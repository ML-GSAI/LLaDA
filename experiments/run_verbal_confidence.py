"""
Experiment: Verbalized Confidence for LLaDA (Xiong et al., 2024 style)

Implements two verbalization approaches for masked diffusion models:

1. Verbalized Confidence (VC): Prompt asks for answer + confidence percentage in one pass.
   Parse the generated confidence score as uncertainty signal.

2. P(True) Elicitation: Generate answer (greedy), then ask
   "Is this correct? (Yes/No)" — extract p(Yes) from the model logits
   at the masked position (analogous to Kadavath et al. 2022).

Compared against:
- Mean denoising entropy (trajectory UQ, Exp 1)
- Semantic entropy (Exp 2)
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
import string
import json
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ─── Utilities ────────────────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t): return ' '.join(t.split())
    def remove_punc(t):
        return ''.join(ch for ch in t if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match(prediction, aliases):
    pred = normalize_answer(prediction)
    return any(normalize_answer(a) == pred or normalize_answer(a) in pred for a in aliases)

def mmlu_match(prediction, gold_letter, gold_choice):
    pred = prediction.strip()
    m = re.search(r'\b([ABCD])\b', pred, flags=re.IGNORECASE)
    if m and m.group(1).upper() == gold_letter:
        return True
    pred_norm = normalize_answer(pred)
    choice_norm = normalize_answer(gold_choice)
    return choice_norm == pred_norm or choice_norm in pred_norm

def compute_ece(confidences, correctness, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            ece += mask.mean() * abs(confidences[mask].mean() - correctness[mask].mean())
    return ece


# ─── LLaDA generation helpers ─────────────────────────────────────────────────

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps,
                                      device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate_greedy(model, prompt, attention_mask=None,
                    steps=64, gen_length=64, mask_id=126336):
    """Standard greedy (temperature=0) generation."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)
    prompt_len = prompt.shape[1]
    block_mask = (x[:, prompt_len:] == mask_id)
    num_transfer = get_num_transfer_tokens(block_mask, steps)

    for i in range(steps):
        mask_index = (x == mask_id)
        logits = model(x, attention_mask=attention_mask).logits
        x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p[:, prompt_len + gen_length:] = -torch.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        _, sel = torch.topk(confidence[0], k=num_transfer[0, i])
        transfer_index[0, sel] = True
        x[transfer_index] = x0[transfer_index]
    return x


@torch.no_grad()
def generate_greedy_ar(model, prompt, attention_mask=None,
                       gen_length=64, pad_token_id=None):
    input_ids = prompt.clone()
    if attention_mask is not None:
        attn = attention_mask.clone().bool()
    else:
        attn = torch.ones_like(prompt, dtype=torch.bool)

    for _ in range(gen_length):
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=torch.bool, device=attn.device)], dim=1)

    return input_ids

@torch.no_grad()
def get_yes_probability(model, prompt_ids, attention_mask,
                        yes_id, no_id, mask_id=126336):
    """
    Append a single [MASK] token, run one model forward pass,
    return p(Yes) / (p(Yes) + p(No)) at that masked position.
    This implements P(True) elicitation for masked diffusion models.
    """
    # Append one MASK token
    x = torch.cat([
        prompt_ids,
        torch.tensor([[mask_id]], dtype=torch.long, device=model.device)
    ], dim=1)
    attn = torch.cat([
        attention_mask,
        torch.ones((1, 1), dtype=attention_mask.dtype, device=model.device)
    ], dim=1)
    logits = model(x, attention_mask=attn).logits  # (1, L+1, V)
    last_logits = logits[0, -1]  # (V,) — the MASK position
    p_yes = torch.softmax(last_logits[[yes_id, no_id]], dim=0)[0].item()
    return p_yes


@torch.no_grad()
def get_yes_probability_ar(model, prompt_ids, attention_mask, yes_id, no_id):
    attn = attention_mask.bool() if attention_mask is not None else None
    logits = model(input_ids=prompt_ids, attention_mask=attn).logits
    next_logits = logits[0, -1]
    p_yes = torch.softmax(next_logits[[yes_id, no_id]], dim=0)[0].item()
    return p_yes


# ─── Parse verbalized confidence from generated text ─────────────────────────

def parse_confidence(text):
    """
    Extract confidence percentage from generated text.
    Looks for patterns like "Confidence: 75%" or "75%" or "75/100".
    Returns float in [0,1], or None if unparseable.
    """
    # Try "X%" pattern
    matches = re.findall(r'(\d{1,3})\s*%', text)
    if matches:
        val = int(matches[-1])
        return min(max(val, 0), 100) / 100.0
    # Try "X/100" pattern
    matches = re.findall(r'(\d{1,3})\s*/\s*100', text)
    if matches:
        val = int(matches[-1])
        return min(max(val, 0), 100) / 100.0
    # Try standalone number 0-100
    matches = re.findall(r'\b(\d{1,3})\b', text)
    for m in reversed(matches):
        val = int(m)
        if 0 <= val <= 100:
            return val / 100.0
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model: {args.model_name}")
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

    def pick_single_token_id(primary, fallback):
        cand = tokenizer.encode(primary, add_special_tokens=False)
        if len(cand) == 1:
            return cand[0]
        cand = tokenizer.encode(fallback, add_special_tokens=False)
        return cand[0]

    # Token IDs for Yes/No
    yes_id = pick_single_token_id(" Yes", "Yes")
    no_id  = pick_single_token_id(" No", "No")
    print(f"Yes token id: {yes_id}, No token id: {no_id}")

    if args.dataset == 'triviaqa':
        print(f"Loading TriviaQA (n={args.n_examples})")
        dataset = load_dataset('trivia_qa', 'rc',
                               split=f'{args.data_split}[:{args.n_examples}]',
                               trust_remote_code=True)
    elif args.dataset == 'mmlu':
        print(f"Loading MMLU (split={args.data_split}, n={args.n_examples})")
        dataset = load_dataset('cais/mmlu', 'all',
                               split=f'{args.data_split}[:{args.n_examples}]',
                               trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    results = []

    for idx, sample in enumerate(tqdm(dataset, desc='Evaluating')):
        question = sample['question']
        if args.dataset == 'triviaqa':
            aliases = sample['answer']['aliases'] + [sample['answer']['value']]
            gold_letter = None
            gold_choice = None
            ans_content = f"Answer the following question in one short phrase.\n\nQuestion: {question}"
        else:
            choices = sample['choices']
            letters = ['A', 'B', 'C', 'D']
            gold_idx = int(sample['answer'])
            gold_letter = letters[gold_idx]
            gold_choice = choices[gold_idx]
            options = "\n".join([f"{letters[i]}. {choices[i]}" for i in range(len(choices))])
            aliases = [gold_choice]
            ans_content = (
                "Answer the following multiple-choice question by giving only the option letter "
                "(A, B, C, or D).\n\n"
                f"Question: {question}\n"
                f"Options:\n{options}"
            )

        # ── Step 1: Generate greedy answer ──────────────────────────────────
        ans_msg = {"role": "user", "content": ans_content}
        ans_prompt = tokenizer.apply_chat_template(
            [ans_msg], add_generation_prompt=True, tokenize=False)
        enc = tokenizer(ans_prompt, add_special_tokens=False, return_tensors='pt')
        model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
        ans_ids = enc['input_ids'].to(model_device)
        ans_attn = enc['attention_mask'].to(model_device)

        if is_llada:
            out = generate_greedy(model, ans_ids, ans_attn,
                                  steps=args.steps, gen_length=args.gen_length,
                                  mask_id=mask_id)
        else:
            out = generate_greedy_ar(model, ans_ids, ans_attn,
                                     gen_length=args.gen_length,
                                     pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(out[0, ans_ids.shape[1]:],
                                  skip_special_tokens=True).strip()
        if args.dataset == 'triviaqa':
            correct = exact_match(answer, aliases)
        else:
            correct = mmlu_match(answer, gold_letter, gold_choice)

        # ── Method 1: P(True) via single-token logit ─────────────────────────
        ptrue_prompt = (
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n"
            f"Is the proposed answer correct? Please answer Yes or No."
        )
        ptrue_msg = {"role": "user", "content": ptrue_prompt}
        ptrue_str = tokenizer.apply_chat_template(
            [ptrue_msg], add_generation_prompt=True, tokenize=False)
        enc2 = tokenizer(ptrue_str, add_special_tokens=False, return_tensors='pt')
        if is_llada:
            p_true = get_yes_probability(
                model,
                enc2['input_ids'].to(model_device),
                enc2['attention_mask'].to(model_device),
                yes_id, no_id, mask_id
            )
        else:
            p_true = get_yes_probability_ar(
                model,
                enc2['input_ids'].to(model_device),
                enc2['attention_mask'].to(model_device),
                yes_id, no_id
            )

        # ── Method 2: Verbalized confidence (answer + % in one generation) ───
        if args.dataset == 'triviaqa':
            vc_content = (
                f"Answer the following question, then give your confidence as a percentage.\n\n"
                f"Question: {question}\n\n"
                f"Format your response exactly as:\n"
                f"Answer: [your answer]\nConfidence: [0-100]%"
            )
        else:
            vc_content = (
                f"Answer the following multiple-choice question, then give your confidence.\n\n"
                f"Question: {question}\n"
                f"Options:\n{options}\n\n"
                f"Format your response exactly as:\n"
                f"Answer: [A/B/C/D]\nConfidence: [0-100]%"
            )
        vc_msg = {"role": "user", "content": vc_content}
        vc_prompt = tokenizer.apply_chat_template(
            [vc_msg], add_generation_prompt=True, tokenize=False)
        enc3 = tokenizer(vc_prompt, add_special_tokens=False, return_tensors='pt')
        vc_ids = enc3['input_ids'].to(model_device)
        vc_attn = enc3['attention_mask'].to(model_device)

        if is_llada:
            vc_out = generate_greedy(model, vc_ids, vc_attn,
                                     steps=args.steps, gen_length=args.vc_gen_length,
                                     mask_id=mask_id)
        else:
            vc_out = generate_greedy_ar(model, vc_ids, vc_attn,
                                        gen_length=args.vc_gen_length,
                                        pad_token_id=tokenizer.eos_token_id)
        vc_text = tokenizer.decode(vc_out[0, vc_ids.shape[1]:],
                                   skip_special_tokens=True).strip()

        # Parse confidence from VC generation
        vc_conf = parse_confidence(vc_text)
        if vc_conf is None:
            vc_conf = 0.5  # default to uncertain if unparseable

        results.append({
            'idx':        idx,
            'question':   question,
            'answer':     answer,
            'vc_text':    vc_text,
            'aliases':    aliases[:3],
            'correct':    int(correct),
            'p_true':     float(p_true),
            'vc_conf':    float(vc_conf),
            'vc_parseable': parse_confidence(vc_text) is not None,
        })

        if (idx + 1) % 10 == 0:
            acc = np.mean([r['correct'] for r in results])
            avg_ptrue = np.mean([r['p_true'] for r in results])
            avg_vc = np.mean([r['vc_conf'] for r in results])
            parse_rate = np.mean([r['vc_parseable'] for r in results])
            print(f"  [{idx+1}/{args.n_examples}] acc={acc:.3f}  "
                  f"avg_p_true={avg_ptrue:.3f}  avg_vc={avg_vc:.3f}  "
                  f"vc_parse_rate={parse_rate:.2f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    correctness = np.array([r['correct'] for r in results])
    print(f"\nFinal accuracy: {correctness.mean():.3f}")

    parse_rate = np.mean([r['vc_parseable'] for r in results])
    print(f"VC parse rate: {parse_rate:.2f}")

    metrics = {
        'p_true':  np.array([r['p_true']  for r in results]),
        'vc_conf': np.array([r['vc_conf'] for r in results]),
    }

    eval_results = {}
    print(f"\n  {'Metric':<22} {'AUROC':>8} {'ECE':>8}")
    print("  " + "-"*40)
    for name, scores in metrics.items():
        auroc = roc_auc_score(correctness, scores)
        ece   = compute_ece(scores, correctness)
        eval_results[name] = {'auroc': float(auroc), 'ece': float(ece)}
        print(f"  {name:<22} {auroc:>8.4f} {ece:>8.4f}")

    # Compare with previous experiments
    print(f"\n  [Previous results for reference]")
    print(f"  {'sem_entropy':<22} {'0.8218':>8} {'0.2462':>8}")
    print(f"  {'mean_entropy(traj)':<22} {'0.7910':>8} {'0.1057':>8}")
    print(f"  {'self_consistency':<22} {'0.7906':>8} {'0.3877':>8}")

    output = {
        'args':         vars(args),
        'accuracy':     float(correctness.mean()),
        'n_correct':    int(correctness.sum()),
        'n_total':      len(results),
        'vc_parse_rate': float(parse_rate),
        'eval_results': eval_results,
        'samples':      results,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',    default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--model_family', choices=['auto', 'llada', 'ar'], default='auto')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--device_map', choices=['none', 'auto'], default='none')
    parser.add_argument('--dataset', choices=['triviaqa', 'mmlu'], default='triviaqa')
    parser.add_argument('--data_split', default='validation')
    parser.add_argument('--n_examples',    type=int, default=200)
    parser.add_argument('--steps',         type=int, default=64)
    parser.add_argument('--gen_length',    type=int, default=32,
                        help='Answer generation length')
    parser.add_argument('--vc_gen_length', type=int, default=64,
                        help='Verbalized confidence generation length (longer for answer+%)' )
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--output',        default='results/triviaqa_verbal_conf.json')
    args = parser.parse_args()
    main(args)
