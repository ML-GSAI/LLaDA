import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def rows_contain_token_sequence(tokens, sequence):
    """Return a boolean flag for each row that contains ``sequence``."""
    sequence_length = sequence.numel()
    if sequence_length == 0 or tokens.shape[1] < sequence_length:
        return torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
    return (tokens.unfold(1, sequence_length, 1) == sequence).all(-1).any(-1)


def get_next_sequence_token_ids(tokens, positions, sequence, context_start):
    """Return the next sequence token for every row at ``positions``."""
    token_ids = sequence[0].expand(tokens.shape[0]).clone()
    matched = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)

    for prefix_length in range(sequence.numel() - 1, 0, -1):
        offsets = torch.arange(-prefix_length, 0, device=tokens.device)
        window = (positions.unsqueeze(1) + offsets).clamp_(min=0)
        fits = (positions - context_start >= prefix_length) & (
            tokens.gather(1, window) == sequence[:prefix_length]
        ).all(dim=1)
        newly_matched = fits & ~matched
        token_ids = torch.where(newly_matched, sequence[prefix_length], token_ids)
        matched |= newly_matched

    return token_ids


def apply_end_think_logit_boost(logits, tokens, candidate_mask_index, context_start,
                                total_gen_length, end_think_token_ids=None,
                                end_think_logit_boost=0., end_think_boost_power=2.):
    """Gradually encourage an end-of-thinking token sequence during generation."""
    if end_think_logit_boost <= 0 or not end_think_token_ids or total_gen_length <= 0:
        return logits
    if any(token_id < 0 or token_id >= logits.shape[-1] for token_id in end_think_token_ids):
        return logits

    sequence = torch.tensor(end_think_token_ids, dtype=torch.long, device=logits.device)
    has_candidate = candidate_mask_index.any(dim=1)
    has_end_think = rows_contain_token_sequence(tokens[:, context_start:], sequence)
    active = has_candidate & ~has_end_think
    if not bool(active.any()):
        return logits

    positions = candidate_mask_index.int().argmax(dim=1)
    token_ids = get_next_sequence_token_ids(
        tokens, positions, sequence, context_start
    )
    progress = (
        ((positions - context_start + 1).to(torch.float64) / float(total_gen_length))
        .clamp_(0.0, 1.0)
    )
    boost = end_think_logit_boost * progress.pow(end_think_boost_power)

    active_rows = torch.nonzero(active, as_tuple=True)[0]
    boosted_logits = logits.clone()
    boosted_logits[
        active_rows, positions[active_rows], token_ids[active_rows]
    ] += boost[active_rows].to(boosted_logits.dtype)
    return boosted_logits


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
             confidence_eos_eot_inf=False, end_think_token_ids=None, end_think_logit_boost=0.,
             end_think_boost_power=2., end_think_context_start=None,
             end_think_total_gen_length=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    base_model = getattr(model, 'module', model)
    model_name = getattr(getattr(base_model, 'config', None), '_name_or_path', '')
    if 'illada' in model_name.lower():
        assert prompt.shape[0] == 1, 'iLLaDA currently does not support padded batch generation.'

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    if end_think_context_start is None:
        end_think_context_start = prompt.shape[1]
    if end_think_total_gen_length is None:
        end_think_total_gen_length = gen_length

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length
            candidate_mask_index = mask_index.clone()
            candidate_mask_index[:, :block_start] = False
            candidate_mask_index[:, block_end:] = False
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits = apply_end_think_logit_boost(
                logits,
                x,
                candidate_mask_index,
                end_think_context_start,
                end_think_total_gen_length,
                end_think_token_ids=end_think_token_ids,
                end_think_logit_boost=end_think_logit_boost,
                end_think_boost_power=end_think_boost_power,
            )

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def var_generate(model, tokenizer, prompt, steps, gen_length, block_length,
                 temperature=0., cfg_scale=0., remasking='low_confidence',
                 mask_id=5, stop_tokens=None, end_think_token_ids=None,
                 end_think_logit_boost=0., end_think_boost_power=2.):
    """Generate one block at a time without adding future mask blocks."""
    assert gen_length % block_length == 0

    x = prompt.clone()
    stop_tokens = stop_tokens or []
    for _ in range(gen_length // block_length):
        x = generate(
            model, x, steps=steps, gen_length=block_length,
            block_length=block_length, temperature=temperature,
            cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
            end_think_token_ids=end_think_token_ids,
            end_think_logit_boost=end_think_logit_boost,
            end_think_boost_power=end_think_boost_power,
            end_think_context_start=prompt.shape[1],
            end_think_total_gen_length=gen_length,
        )
        text = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=False)
        if any(stop_token in text for stop_token in stop_tokens):
            break
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
