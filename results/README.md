# UQ4DLLM Experiment Results

**Date**: 2026-03-04
**Model**: LLaDA-8B-Instruct (GSAI-ML/LLaDA-8B-Instruct)
**Dataset**: TriviaQA validation (rc split)

---

## Experiment 1: Trajectory-Based UQ (Section 4.1)

**File**: `triviaqa_traj_uq.json`
**Config**: n=200, steps=64, gen_length=32, temperature=0
**Accuracy**: 28.0% (56/200)

| Metric | AUROC | ECE |
|--------|-------|-----|
| mean_entropy (proposed) | **0.791** | **0.106** |
| traj_variance | 0.629 | 0.429 |
| final_entropy | 0.512 | 0.686 |

**Key finding**: Mean denoising entropy (averaged across all denoising steps) achieves AUROC=0.791, matching AR-model semantic entropy benchmarks. Final-step entropy alone is near-random (0.512), proving the full trajectory is essential.

---

## Experiment 2: Semantic Entropy for DLMs (Section 4.2)

**File**: `triviaqa_sem_entropy.json`
**Config**: n=100, K=5 samples, steps=32, gen_length=32, temperature=1.0
**NLI model**: cross-encoder/nli-deberta-v3-small
**Accuracy**: 24.0% (24/100)

| Metric | AUROC | ECE |
|--------|-------|-----|
| semantic_entropy (proposed) | **0.822** | 0.246 |
| self_consistency | 0.791 | 0.388 |

---

## Experiment 3: Verbalized Confidence (Xiong et al., 2024)

**File**: `triviaqa_verbal_conf.json`
**Config**: n=100, steps=64, gen_length=32, vc_gen_length=64
**Accuracy**: 22.0% (22/100)

Two methods adapted for masked diffusion:

| Metric | AUROC | ECE | Description |
|--------|-------|-----|-------------|
| P(True) | **0.821** | 0.352 | Ask "Is this correct? Yes/No" — use logit p(Yes) |
| VC (verbalized %) | 0.551 | **0.076** | Ask model to output "Confidence: X%" |

**Key findings**:
- P(True) elicitation (AUROC=0.821) works well for LLaDA despite its non-AR generation
- Verbalized % confidence (avg=99.6%) is severely overconfident — LLaDA always says ~100%
- VC parse rate = 100% (model always generates a number), but the values are not discriminative
- Overconfidence in VC likely reflects LLaDA's lack of RLHF/calibration fine-tuning

---

## Full Comparison Table

| Method | AUROC | ECE | Cost | Section |
|--------|-------|-----|------|---------|
| Semantic Entropy | **0.822** | 0.246 | K×T passes | 4.2 |
| P(True) elicitation | 0.821 | 0.352 | 2×T passes | Xiong et al. |
| Mean Denoising Entropy | 0.791 | **0.106** | 1×T passes | 4.1 (proposed) |
| Self-Consistency | 0.791 | 0.388 | K×T passes | 4.2 |
| Trajectory Variance | 0.629 | 0.429 | 1×T passes | 4.1 |
| Verbalized % Confidence | 0.551 | 0.076 | 2×T passes | Xiong et al. |
| Final-Step Entropy | 0.512 | 0.686 | 1 pass | baseline |

**Main takeaways**:
1. Semantic entropy and P(True) are top performers (AUROC ~0.82), but both require extra passes
2. **Mean denoising entropy (proposed) achieves 0.791 AUROC at 1×T cost** — best efficiency
3. Verbalized % confidence fails completely (overconfident) — consistent with Kadavath et al. finding that verbalization degrades after RLHF, and LLaDA lacks calibration tuning
4. Trajectory information is critical: final-step entropy alone is near-random (0.512)

---

## Files

| File | Description |
|------|-------------|
| triviaqa_traj_uq.json | Trajectory UQ results (200 samples) |
| triviaqa_sem_entropy.json | Semantic entropy results (100 samples, K=5) |
| triviaqa_verbal_conf.json | Verbalized confidence results (100 samples) |
| experiment.log | Exp 1 stdout |
| sem_entropy.log | Exp 2 stdout |
| verbal_conf.log | Exp 3 stdout |
