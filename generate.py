import os
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

@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, padding_id=126081,
                   tokenizer=None, diff_logits_eos_inf=False, diff_confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L), potentially left-padded, or list of chat messages.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        padding_id: The token id for padding is 126081.
        tokenizer: Tokenizer for processing chat messages (optional).
        diff_logits_eos_inf: Whether to set EOS token logits to -inf.
        diff_confidence_eos_eot_inf: Whether to set EOS and EOT token logits to -inf for confidence.
    '''
    batch_size, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    
    # Detect if left-padded or right-padded
    is_left = False
    for i in range(batch_size):
        if (prompt[i, 0] == padding_id) and (prompt[i, -1] != padding_id):
            is_left = True
            break
    
    if not is_left:
        batch_padding_position = []
        for i in range(batch_size):
            padding_pos = (prompt[i] == padding_id).nonzero(as_tuple=True)[0]
            if len(padding_pos) > 0:
                batch_padding_position.append(padding_pos[0].item())
            else:
                batch_padding_position.append(prompt_len)
        
        # Where padding starts
        prompt_len = batch_padding_position.copy()

    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=model.device)
    if not is_left:
        for i in range(batch_size):
            x[i, :prompt_len[i]] = prompt[i, :prompt_len[i]].clone()
            x[i, prompt_len[i] + gen_length:] = torch.full((total_len - (prompt_len[i] + gen_length),), padding_id, dtype=torch.long) 
    else:
        x[:, :prompt_len] = prompt.clone()

    original_prompt_mask = (prompt != padding_id)
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    if not is_left:
        for i in range(batch_size):
            prompt_index[i, :prompt_len[i]] = original_prompt_mask[i, :prompt_len[i]]
    else:
        prompt_index[:, :prompt_len] = original_prompt_mask

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    attention_mask = torch.ones((batch_size, total_len), device=x.device, dtype=torch.long)
    
    if not is_left:
        for i in range(batch_size):
            attention_mask[i, prompt_len[i] + gen_length:] = 0 
    else:
        attention_mask[:, :prompt_len] = original_prompt_mask.long()
    
    for num_block in range(num_blocks):
        if not is_left:
            start_pos = [prompt_len[i] + num_block * block_length for i in range(batch_size)]
            end_pos = [prompt_len[i] + (num_block + 1) * block_length for i in range(batch_size)]
            block_mask_index = torch.zeros((batch_size, block_length), dtype=torch.bool, device=x.device)
            for i in range(batch_size):
                block_mask_index[i] = (x[i, start_pos[i]:end_pos[i]] == mask_id)
        else:
            start_pos = prompt_len + num_block * block_length
            end_pos = prompt_len + (num_block + 1) * block_length
            block_mask_index = (x[:, start_pos:end_pos] == mask_id)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            if not mask_index.any():
                break
                
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                
                x_ = torch.cat([x, un_x], dim=0)
                extended_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                
                logits = model(x_, attention_mask=extended_attention_mask).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            
            if diff_logits_eos_inf:
                logits[:, :, 126081] = -torch.inf
            
            if temperature > 0.:
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
            else:
                x0 = torch.argmax(logits, dim=-1)
            
            if diff_confidence_eos_eot_inf:
                logits[:, :, 126081] = -torch.inf
                logits[:, :, 126348] = -torch.inf
            
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
            elif remasking == 'random':
                x0_p = torch.rand_like(logits[:, :, 0])
            else:
                raise NotImplementedError(remasking)

            confidence_mask = torch.ones_like(x, dtype=torch.bool)
            if not is_left:
                for j in range(batch_size):
                    confidence_mask[j, end_pos[j]:] = False
            else:
                confidence_mask[:, end_pos:] = False
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index & confidence_mask, x0_p, -torch.inf)

            num_masked_in_block = block_mask_index.sum(dim=-1)
            num_to_reveal = torch.ceil(num_masked_in_block * (1.0 / steps_per_block)).long()

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = min(num_to_reveal[j].item(), (confidence[j] > -torch.inf).sum().item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            
    if not is_left:
        for i in range(batch_size):
            x_generated = x[i, prompt_len[i]:prompt_len[i] + gen_length].clone()
            x[i, -gen_length:] = x_generated.clone()
            x[i, :-gen_length] = prompt[i].clone()
    
    # Decode if tokenizer is provided
    if tokenizer is not None:
        responses = []
        for i in range(batch_size):
            responses.append(tokenizer.decode(x[i, -gen_length:], skip_special_tokens=True))
        print('--------------------')
        for i in range(batch_size):
            print(f'Response {i}:', responses[i])
            print('====================')
        print('--------------------')
        return responses
    
    return x
def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('/home/xushaoxuan/huggingface/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/home/xushaoxuan/huggingface/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
               "Simplify $\\sqrt{242}$.\nPlease put your final answer within \\boxed{}.\n"]
    # print(prompt)
    # Add special tokens for the Instruction model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt[i]} for i in range(len(prompt))]
    # print(m)
    prompt = [tokenizer.apply_chat_template([mi], add_generation_prompt=True, tokenize=False) for mi in m]
    # print(prompt)
    ## batch generation
    input_ids = tokenizer.batch_encode_plus(prompt, padding = True)['input_ids']
    input_ids = torch.tensor(input_ids).to(device)
    batch_out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    batch_tokens = tokenizer.batch_decode(batch_out[:, input_ids.shape[1]:], skip_special_tokens=True)
    # single generation
    single_tokens = []
    input_ids = tokenizer.batch_encode_plus(prompt, padding = False)['input_ids']
    for input_id in input_ids:
        input_id = torch.tensor(input_id).to(device).unsqueeze(0)
        single_out = generate(model, input_id, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
        single_tokens.append(tokenizer.batch_decode(single_out[:, input_id.shape[1]:], skip_special_tokens=True)[0])
    for i in range(len(prompt)):
        print(f'Prompt {i+1}: {prompt[i]}')
        print(f'Batch Generation {i+1}: {batch_tokens[i]}')
        print(f'Single Generation {i+1}: {single_tokens[i]}')
        print('-----------------------------------------------------------------------')
if __name__ == '__main__':
    main()
