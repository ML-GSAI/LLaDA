"""
Experiment: Trajectory-Based Uncertainty Quantification for LLaDA on TriviaQA

Implements Section 4.1 of the proposal:
- Token-level denoising entropy: H_t^i = -sum_v p(x0^i=v | x_t) log p(x0^i=v | x_t)
- Trajectory variance: how much argmax predictions change across denoising steps
- Final-step entropy: entropy at the last denoising step (baseline)

Evaluation:
- AUROC: uncertainty score vs binary correctness (hallucination detection)
- ECE: predicted confidence vs actual accuracy
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
import json
import re
import string
import argparse
from tqdm import tqdm


# ─── Utility: Answer normalization ───────────────────────────────────────────

def normalize_answer(s):
    """Lower text, remove punctuation/articles/extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction, ground_truth_aliases):
    pred_norm = normalize_answer(prediction)
    for alias in ground_truth_aliases:
        if normalize_answer(alias) == pred_norm or normalize_answer(alias) in pred_norm:
            return True
    return False


def mmlu_match(prediction, gold_letter, gold_choice):
    pred = prediction.strip()
    m = re.search(r'\b([ABCD])\b', pred, flags=re.IGNORECASE)
    if m and m.group(1).upper() == gold_letter:
        return True
    pred_norm = normalize_answer(pred)
    choice_norm = normalize_answer(gold_choice)
    return choice_norm == pred_norm or choice_norm in pred_norm


# ─── Utility: ECE computation ────────────────────────────────────────────────

def compute_ece(confidences, correctness, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            ece += mask.mean() * abs(bin_conf - bin_acc)
    return ece


# ─── Modified generate with trajectory capture ───────────────────────────────

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (torch.zeros(mask_num.size(0), steps,
                                       device=mask_index.device, dtype=torch.int64) + base)
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_with_trajectory(model, prompt, attention_mask=None,
                              steps=64, gen_length=64, block_length=64,
                              temperature=0., mask_id=126336):
    """
    Modified generate that captures per-step logits for uncertainty estimation.

    Returns:
        x: final generated tokens (1, prompt_len + gen_length)
        trajectory: list of dicts with per-step entropy and argmax predictions
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    prompt_len = prompt.shape[1]
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    trajectory = []  # one entry per denoising step

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end   = prompt_len + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)  # (1, L)

            logits = model(x, attention_mask=attention_mask).logits  # (1, L, V)

            # ── Uncertainty signals ──────────────────────────────────────────
            probs = F.softmax(logits[0, prompt_len:], dim=-1)  # (gen_length, V)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)  # (gen_length,)
            argmax_pred = torch.argmax(logits[0, prompt_len:], dim=-1)  # (gen_length,)
            is_masked = mask_index[0, prompt_len:].cpu()  # (gen_length,) bool

            trajectory.append({
                'step': num_block * steps_per_block + i,
                'entropy': entropy.float().cpu().numpy(),           # (gen_length,)
                'argmax': argmax_pred.cpu().numpy(),        # (gen_length,)
                'masked': is_masked.numpy(),                # (gen_length,) bool
            })

            # ── Standard low-confidence remasking step ───────────────────────
            x0 = torch.argmax(logits, dim=-1)  # (1, L)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)  # (1, L)

            x0_p[:, block_end:] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p,
                                     torch.full_like(x0_p, -torch.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            _, sel = torch.topk(confidence[0], k=num_transfer_tokens[0, i])
            transfer_index[0, sel] = True
            x[transfer_index] = x0[transfer_index]

    return x, trajectory


@torch.no_grad()
def generate_with_trajectory_ar(model, prompt, attention_mask=None,
                                gen_length=64, temperature=0.0):
    """
    AR generation with per-step entropy trajectory.
    """
    input_ids = prompt.clone()
    if attention_mask is not None:
        attn = attention_mask.clone().bool()
    else:
        attn = torch.ones_like(prompt, dtype=torch.bool)
    trajectory = []
    past_key_values = None
    cache_supported = True

    for step in range(gen_length):
        if not cache_supported:
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
            )
        elif past_key_values is None:
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=True,
            )
        else:
            out = model(
                input_ids=input_ids[:, -1:],
                attention_mask=attn,
                past_key_values=past_key_values,
                use_cache=True,
            )

        if cache_supported and hasattr(out, "past_key_values") and out.past_key_values is not None:
            past_key_values = out.past_key_values
        else:
            cache_supported = False
            past_key_values = None
        logits = out.logits[:, -1, :]  # (1, V)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)
        argmax_pred = torch.argmax(logits, dim=-1)

        trajectory.append({
            'step': step,
            'entropy': np.array([float(entropy.item())], dtype=np.float32),
            'argmax': np.array([int(argmax_pred.item())], dtype=np.int64),
            'masked': np.array([True], dtype=bool),
        })

        if temperature > 0:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = argmax_pred.unsqueeze(-1)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=torch.bool, device=attn.device)], dim=1)

    return input_ids, trajectory


# ─── Uncertainty metric aggregation ──────────────────────────────────────────

def aggregate_trajectory(trajectory, gen_length):
    """
    From the full trajectory, compute per-sequence scalar uncertainty metrics.

    Returns dict with keys:
        mean_entropy        – mean denoising entropy over all (step, masked position) pairs
        final_entropy       – mean entropy at the last denoising step
        trajectory_variance – fraction of positions whose argmax changes across steps
        mean_confidence     – 1 - mean_entropy (proxy for confidence, normalized)
    """
    all_entropies = []     # entropy of masked positions at each step
    argmax_matrix = []     # shape (steps, gen_length)

    for step_data in trajectory:
        ent   = step_data['entropy']      # (gen_length,)
        mask  = step_data['masked']       # bool (gen_length,)
        argmax_matrix.append(step_data['argmax'])
        if mask.any():
            all_entropies.append(ent[mask].mean())

    mean_entropy   = float(np.mean(all_entropies)) if all_entropies else 0.0
    final_entropy  = float(trajectory[-1]['entropy'].mean())

    # Trajectory variance: for each position, does argmax change?
    argmax_mat = np.stack(argmax_matrix, axis=0)  # (steps, gen_length)
    # fraction of positions with >1 distinct argmax across steps
    n_unique_per_pos = np.array([len(np.unique(argmax_mat[:, j]))
                                 for j in range(gen_length)])
    traj_variance = float((n_unique_per_pos > 1).mean())

    # Normalize entropy to [0,1] proxy confidence (log V is max entropy for vocab size)
    # We'll just use raw entropy; AUROC doesn't need normalization
    return {
        'mean_entropy':       mean_entropy,
        'final_entropy':      final_entropy,
        'traj_variance':      traj_variance,
        'mean_confidence':    1.0 / (1.0 + mean_entropy),  # monotone inverse
    }


def aggregate_trajectory_ar(trajectory):
    entropies = [float(step_data['entropy'][0]) for step_data in trajectory]
    argmaxes = [int(step_data['argmax'][0]) for step_data in trajectory]

    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    final_entropy = float(entropies[-1]) if entropies else 0.0
    if len(argmaxes) <= 1:
        traj_variance = 0.0
    else:
        changes = np.array(argmaxes[1:]) != np.array(argmaxes[:-1])
        traj_variance = float(changes.mean())

    return {
        'mean_entropy':       mean_entropy,
        'final_entropy':      final_entropy,
        'traj_variance':      traj_variance,
        'mean_confidence':    1.0 / (1.0 + mean_entropy),
    }


# ─── Main experiment ─────────────────────────────────────────────────────────

def main(args):
    print(f"Loading model: {args.model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_llada = args.model_family == 'llada' or (args.model_family == 'auto' and 'llada' in args.model_name.lower())

    use_trust_remote_code = is_llada or args.trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=use_trust_remote_code)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    load_kwargs = {'torch_dtype': model_dtype}
    if args.device_map == 'auto':
        load_kwargs['device_map'] = 'auto'
    if is_llada:
        model = AutoModel.from_pretrained(
            args.model_name, trust_remote_code=True, **load_kwargs
        ).eval()
        mask_id = args.mask_id
        assert tokenizer.pad_token_id != mask_id
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, trust_remote_code=use_trust_remote_code, **load_kwargs
            ).eval()
        except ValueError:
            model = AutoModel.from_pretrained(
                args.model_name, trust_remote_code=use_trust_remote_code, **load_kwargs
            ).eval()
        mask_id = None

    if args.dataset == 'triviaqa':
        print(f"Loading TriviaQA (n={args.n_examples})")
        dataset = load_dataset('trivia_qa', 'rc', split=f'{args.data_split}[:{args.n_examples}]',
                               trust_remote_code=True)
    elif args.dataset == 'mmlu':
        print(f"Loading MMLU (split={args.data_split}, n={args.n_examples})")
        dataset = load_dataset('cais/mmlu', 'all', split=f'{args.data_split}[:{args.n_examples}]',
                               trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    results = []

    for idx, sample in enumerate(tqdm(dataset, desc='Evaluating')):
        question = sample['question']
        if args.dataset == 'triviaqa':
            aliases = sample['answer']['aliases'] + [sample['answer']['value']]
            prompt_content = f"Answer the following question concisely.\n\nQuestion: {question}\nAnswer:"
            gold_letter = None
            gold_choice = None
        else:
            choices = sample['choices']
            letters = ['A', 'B', 'C', 'D']
            gold_idx = int(sample['answer'])
            gold_letter = letters[gold_idx]
            gold_choice = choices[gold_idx]
            options = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(len(choices))])
            aliases = [gold_choice]
            prompt_content = (
                "Answer the following multiple-choice question by giving only the option letter "
                "(A, B, C, or D).\n\n"
                f"Question: {question}\n"
                f"Options:\n{options}\n"
                "Answer:"
            )

        # Format prompt
        message  = {"role": "user", "content": prompt_content}
        prompt_str = tokenizer.apply_chat_template(
            [message], add_generation_prompt=True, tokenize=False)

        enc = tokenizer(prompt_str, add_special_tokens=False,
                        return_tensors='pt')
        model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
        input_ids = enc['input_ids'].to(model_device)
        attention_mask = enc['attention_mask'].to(model_device)

        # Run generation with trajectory
        if is_llada:
            out, trajectory = generate_with_trajectory(
                model, input_ids, attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.gen_length,  # single block
                temperature=0.,
                mask_id=mask_id
            )
            uq = aggregate_trajectory(trajectory, args.gen_length)
        else:
            out, trajectory = generate_with_trajectory_ar(
                model, input_ids, attention_mask,
                gen_length=args.gen_length,
                temperature=0.0
            )
            uq = aggregate_trajectory_ar(trajectory)

        # Decode
        generated = tokenizer.decode(
            out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Correctness
        if args.dataset == 'triviaqa':
            correct = exact_match(generated, aliases)
        else:
            correct = mmlu_match(generated, gold_letter, gold_choice)

        results.append({
            'idx':        idx,
            'question':   question,
            'generated':  generated,
            'aliases':    aliases[:3],   # save first 3 to keep file small
            'correct':    int(correct),
            **uq,
        })

        if (idx + 1) % 10 == 0:
            acc = np.mean([r['correct'] for r in results])
            print(f"  [{idx+1}/{args.n_examples}] Accuracy so far: {acc:.3f}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    correctness = np.array([r['correct'] for r in results])
    print(f"\nFinal accuracy: {correctness.mean():.3f}")

    metrics = ['mean_entropy', 'final_entropy', 'traj_variance', 'mean_confidence']
    eval_results = {}

    for m in metrics:
        scores = np.array([r[m] for r in results])
        # For entropy/variance: higher = more uncertain = predicts error
        # For confidence: higher = more confident = predicts correct
        if m == 'mean_confidence':
            auroc = roc_auc_score(correctness, scores)
            ece   = compute_ece(scores, correctness)
        else:
            auroc = roc_auc_score(correctness, -scores)  # flip: high entropy → predict wrong
            ece   = compute_ece(1 / (1 + scores), correctness)

        eval_results[m] = {'auroc': float(auroc), 'ece': float(ece)}
        print(f"  {m:20s}  AUROC={auroc:.4f}  ECE={ece:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        'args':         vars(args),
        'accuracy':     float(correctness.mean()),
        'n_correct':    int(correctness.sum()),
        'n_total':      len(results),
        'eval_results': eval_results,
        'samples':      results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--model_family', choices=['auto', 'llada', 'ar'], default='auto')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--device_map', choices=['none', 'auto'], default='none')
    parser.add_argument('--dataset', choices=['triviaqa', 'mmlu'], default='triviaqa')
    parser.add_argument('--data_split', default='validation')
    parser.add_argument('--n_examples',  type=int, default=200)
    parser.add_argument('--steps',       type=int, default=64,
                        help='Denoising steps (used to capture trajectory)')
    parser.add_argument('--gen_length',  type=int, default=32,
                        help='Max answer generation length in tokens')
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--output',      default='results/triviaqa_traj_uq.json')
    args = parser.parse_args()
    main(args)
