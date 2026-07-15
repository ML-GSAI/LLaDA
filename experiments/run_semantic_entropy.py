"""
Experiment: Semantic Entropy for LLaDA on TriviaQA (Section 4.2)

Adapts Kuhn et al. (ICLR 2023) to DLMs:
1. Generate K independent denoising chains from same prompt
2. Cluster outputs by semantic equivalence via NLI
3. Compute semantic entropy: H_sem = -sum_c P(c) log P(c)

Compares against trajectory-based UQ from run_trajectory_uq.py.
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
import json
import re
import string
import argparse
from itertools import combinations
from tqdm import tqdm


# ─── Answer normalization ─────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match(prediction, aliases):
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(a) == pred_norm or normalize_answer(a) in pred_norm
               for a in aliases)

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


# ─── LLaDA generation ─────────────────────────────────────────────────────────

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
def generate_single(model, prompt, attention_mask=None,
                    steps=32, gen_length=32, temperature=1.0, mask_id=126336, seed=None):
    """Single independent generation with optional seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    prompt_len = prompt.shape[1]
    block_mask_index = (x[:, prompt_len:] == mask_id)
    num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

    for i in range(steps):
        mask_index = (x == mask_id)
        logits = model(x, attention_mask=attention_mask).logits

        if temperature > 0:
            logits_f = logits.to(torch.float64)
            noise = torch.rand_like(logits_f)
            gumbel = (-torch.log(noise)) ** temperature
            logits_sampled = logits_f.float().exp() / gumbel.float()
            x0 = torch.argmax(logits_sampled, dim=-1)
        else:
            x0 = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        x0_p[:, prompt_len + gen_length:] = -torch.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        _, sel = torch.topk(confidence[0], k=num_transfer_tokens[0, i])
        transfer_index[0, sel] = True
        x[transfer_index] = x0[transfer_index]

    return x


@torch.no_grad()
def generate_single_ar(model, prompt, attention_mask=None,
                       gen_length=32, temperature=1.0, seed=None, pad_token_id=None):
    if seed is not None:
        torch.manual_seed(seed)
    input_ids = prompt.clone()
    if attention_mask is not None:
        attn = attention_mask.clone().bool()
    else:
        attn = torch.ones_like(prompt, dtype=torch.bool)

    for _ in range(gen_length):
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        if temperature > 0:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=torch.bool, device=attn.device)], dim=1)

    return input_ids


# ─── Semantic clustering via NLI ──────────────────────────────────────────────

class SemanticClusterer:
    def __init__(self, model_name='cross-encoder/nli-deberta-v3-small', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        # label order: contradiction, entailment, neutral (deberta-v3-small)
        # We check the label2id to be safe
        self.entail_idx = self.model.config.label2id.get('entailment', 1)

    @torch.no_grad()
    def are_equivalent(self, s1, s2):
        """True if s1 <-> s2 are semantically equivalent (bidirectional entailment)."""
        pairs = [(s1, s2), (s2, s1)]
        enc = self.tokenizer(
            [p[0] for p in pairs], [p[1] for p in pairs],
            return_tensors='pt', padding=True, truncation=True, max_length=256
        ).to(self.device)
        logits = self.model(**enc).logits  # (2, 3)
        probs = torch.softmax(logits, dim=-1)
        entail_probs = probs[:, self.entail_idx]
        return bool((entail_probs > 0.5).all())

    def cluster(self, texts):
        """Return list of cluster IDs (one per text) via transitive closure."""
        n = len(texts)
        # Build adjacency matrix
        equiv = [[False] * n for _ in range(n)]
        for i in range(n):
            equiv[i][i] = True
        for i, j in combinations(range(n), 2):
            if self.are_equivalent(texts[i], texts[j]):
                equiv[i][j] = equiv[j][i] = True

        # Transitive closure -> cluster IDs
        cluster_ids = [-1] * n
        cid = 0
        for i in range(n):
            if cluster_ids[i] == -1:
                cluster_ids[i] = cid
                for j in range(i+1, n):
                    if equiv[i][j]:
                        cluster_ids[j] = cid
                cid += 1
        return cluster_ids


# ─── Semantic entropy computation ─────────────────────────────────────────────

def semantic_entropy(cluster_ids):
    """H_sem = -sum_c P(c) log P(c)"""
    counts = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    n = len(cluster_ids)
    h = 0.0
    for count in counts.values():
        p = count / n
        h -= p * np.log(p)
    return h

def self_consistency_score(texts):
    """Uncertainty = 1 - (frequency of most common answer)."""
    norm = [normalize_answer(t) for t in texts]
    counts = {}
    for t in norm:
        counts[t] = counts.get(t, 0) + 1
    max_freq = max(counts.values()) / len(texts)
    return 1.0 - max_freq  # high = uncertain


# ─── Main ─────────────────────────────────────────────────────────────────────

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

    print(f"Loading NLI model: {args.nli_model}")
    clusterer = SemanticClusterer(args.nli_model, device=device)

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

        message = {"role": "user", "content": prompt_content}
        prompt_str = tokenizer.apply_chat_template(
            [message], add_generation_prompt=True, tokenize=False)
        enc = tokenizer(prompt_str, add_special_tokens=False, return_tensors='pt')
        model_device = model.device if hasattr(model, "device") else next(model.parameters()).device
        input_ids = enc['input_ids'].to(model_device)
        attn_mask = enc['attention_mask'].to(model_device)

        # Generate K independent samples
        generations = []
        for k in range(args.k_samples):
            if is_llada:
                out = generate_single(model, input_ids, attn_mask,
                                      steps=args.steps, temperature=args.temperature,
                                      gen_length=args.gen_length,
                                      mask_id=mask_id,
                                      seed=idx * 1000 + k)
            else:
                out = generate_single_ar(model, input_ids, attn_mask,
                                         gen_length=args.gen_length,
                                         temperature=args.temperature,
                                         seed=idx * 1000 + k,
                                         pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0, input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
            generations.append(text)

        # Correctness of first (greedy-equivalent) generation
        if args.dataset == 'triviaqa':
            correct = exact_match(generations[0], aliases)
        else:
            correct = mmlu_match(generations[0], gold_letter, gold_choice)

        # Semantic clustering + entropy
        cluster_ids = clusterer.cluster(generations)
        h_sem = semantic_entropy(cluster_ids)
        n_clusters = len(set(cluster_ids))

        # Self-consistency
        sc_uncertainty = self_consistency_score(generations)

        results.append({
            'idx':              idx,
            'question':         question,
            'generations':      generations,
            'aliases':          aliases[:3],
            'correct':          int(correct),
            'sem_entropy':      float(h_sem),
            'n_clusters':       n_clusters,
            'self_consistency': float(sc_uncertainty),
        })

        if (idx + 1) % 10 == 0:
            acc = np.mean([r['correct'] for r in results])
            print(f"  [{idx+1}/{args.n_examples}] acc={acc:.3f}  "
                  f"avg_sem_entropy={np.mean([r['sem_entropy'] for r in results]):.3f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    correctness = np.array([r['correct'] for r in results])
    print(f"\nFinal accuracy: {correctness.mean():.3f}")

    metrics = {
        'sem_entropy':      np.array([r['sem_entropy'] for r in results]),
        'self_consistency': np.array([r['self_consistency'] for r in results]),
    }

    eval_results = {}
    print(f"\n  {'Metric':<22} {'AUROC':>8} {'ECE':>8}")
    print("  " + "-"*40)
    for name, scores in metrics.items():
        auroc = roc_auc_score(correctness, -scores)  # high uncertainty → wrong
        conf = 1 / (1 + scores)
        ece = compute_ece(conf, correctness)
        eval_results[name] = {'auroc': float(auroc), 'ece': float(ece)}
        print(f"  {name:<22} {auroc:>8.4f} {ece:>8.4f}")

    # Compare with trajectory results if available
    try:
        with open(args.traj_results) as f:
            traj = json.load(f)
        print(f"\n  [Trajectory UQ comparison]")
        for m, v in traj['eval_results'].items():
            print(f"  {m:<22} {v['auroc']:>8.4f} {v['ece']:>8.4f}")
    except FileNotFoundError:
        pass

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
    parser.add_argument('--model_name',   default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--model_family', choices=['auto', 'llada', 'ar'], default='auto')
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--device_map', choices=['none', 'auto'], default='none')
    parser.add_argument('--dataset', choices=['triviaqa', 'mmlu'], default='triviaqa')
    parser.add_argument('--data_split', default='validation')
    parser.add_argument('--nli_model',    default='cross-encoder/nli-deberta-v3-small')
    parser.add_argument('--n_examples',   type=int, default=200)
    parser.add_argument('--k_samples',    type=int, default=5,
                        help='Number of independent samples per question')
    parser.add_argument('--steps',        type=int, default=32,
                        help='Denoising steps per sample')
    parser.add_argument('--gen_length',   type=int, default=32)
    parser.add_argument('--temperature',   type=float, default=1.0)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--traj_results', default='results/triviaqa_traj_uq.json')
    parser.add_argument('--output',       default='results/triviaqa_sem_entropy.json')
    args = parser.parse_args()
    main(args)
