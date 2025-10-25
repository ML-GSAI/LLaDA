# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLaDA (Large Language Diffusion with mAsking)** is an 8B-scale diffusion language model that rivals autoregressive models like LLaMA3 8B in performance. Unlike traditional autoregressive models that predict tokens sequentially, LLaDA uses a masked diffusion modeling approach for text generation.

Key papers and resources:
- Paper: https://arxiv.org/abs/2502.09992
- Models: [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base), [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- Related work: RADD, SMDM

## Model Architecture

LLaDA employs a **Transformer Encoder** (identical to Transformer Decoder in parameters, but without causal masking):
- Uses full bidirectional attention instead of causal attention
- Reserved token ID `126336` is used as the [MASK] token
- EOS token ID: `126081`
- EOT token ID: `126348`
- The architecture is based on the LLaMA architecture with the causal mask removed

**Key theoretical insight**: LLaDA uses a variable masking ratio (randomly between 0 and 1) during training, making the training objective an upper bound on negative log-likelihood. This makes LLaDA a true generative model capable of in-context learning and instruction-following.

## Dependencies and Setup

Required transformers version: `transformers==4.38.2`

Basic installation:
```bash
pip install transformers==4.38.2 torch
```

For evaluation with lm-evaluation-harness:
```bash
pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

For OpenCompass evaluation (Instruct models):
```bash
cd opencompass
conda create --name llada_eval python=3.10 -y
conda activate llada_eval
pip install -e .
```

## Inference

### Loading Models

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

### Two Inference Modes

1. **Conditional Generation** (`generate.py`):
   - Used for generating text given a prompt
   - Supports semi-autoregressive generation with `block_length` parameter
   - Key parameters:
     - `steps`: Number of diffusion sampling steps (≤ gen_length)
     - `gen_length`: Length of generated answer
     - `block_length`: Block size for semi-autoregressive generation (≤ gen_length)
     - `temperature`: Sampling temperature (0 for greedy)
     - `cfg_scale`: Classifier-free guidance scale
     - `remasking`: Strategy for remasking ('low_confidence' or 'random')

2. **Conditional Likelihood Estimation** (`get_log_likelihood.py`):
   - Used for computing log-likelihood of a completion given a prompt
   - Uses Monte Carlo estimation
   - Key parameters:
     - `mc_num`: Number of Monte Carlo samples (128 for most tasks, 1 for single-token tasks like MMLU)
     - `batch_size`: Mini-batch size for MC estimation

### Running the Chat Demo

```bash
python chat.py
```

### Running the Gradio Demo

```bash
pip install gradio
python app.py
```

## Evaluation

### Two Evaluation Approaches

1. **lm-evaluation-harness** (for Base model):
   - Supports both conditional likelihood and conditional generation
   - See `eval_llada.sh` for example commands
   - Run with: `accelerate launch eval_llada.py --tasks <task_name> --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',...`

2. **OpenCompass** (for Instruct models):
   - Located in `opencompass/` directory
   - Run with: `bash opencompass/eval_llada_instruct.sh`
   - Custom model paths can be configured in `opencompass/opencompass/configs/models/dllm/`

### Important Evaluation Notes

- **Base model**: Uses conditional likelihood for some metrics (MMLU, HellaSwag, etc.) and generation for others (GSM8K, Math, HumanEval, etc.)
- **Instruct model**: Uses conditional generation for all metrics
- For lm-evaluation-harness bugs with Instruct models, see EVAL.md
- Different cfg_scale values are used for different benchmarks (0.0, 0.5, or 2.0)

## Key Files and Their Purposes

- `generate.py`: Core generation function with diffusion sampling
- `get_log_likelihood.py`: Log-likelihood computation with Monte Carlo estimation
- `chat.py`: Interactive multi-turn chat interface
- `app.py`: Gradio web demo
- `eval_llada.py`: Integration with lm-evaluation-harness
- `eval_llada.sh`: Evaluation commands for various benchmarks
- `GUIDELINES.md`: Model architecture, training, and inference guidelines
- `EVAL.md`: Evaluation instructions and benchmark results

## Training Code Architecture

The repository does NOT include training code, but `GUIDELINES.md` provides:
- Pre-training loss computation (forward diffusion process with variable masking ratio)
- SFT modifications (no noise added to prompts, only to completions)
- Core concept: Simply modify autoregressive training code by replacing causal attention with full attention and adding the masking-based loss

## Important Implementation Details

### Padding
- The sampling code in `generate.py` is implemented for **left-padding**
- Ensure `tokenizer.padding_side = 'left'` for generation
- Padding token ID must NOT equal mask token ID (126336)

### Special Tokens
- Mask token: `126336`
- EOS token: `126081`
- EOT token: `126348`

### Sampling Parameters
- **Semi-autoregressive sampling**: Set `block_length < gen_length` to generate in blocks
- **Full parallel sampling**: Set `block_length = gen_length`
- Optimal performance: `steps = gen_length`
- For faster inference with quality tradeoff: reduce steps (see ablations in EVAL.md)

### Classifier-Free Guidance (CFG)
- Unsupervised CFG is used to improve benchmark performance
- Different tasks use different cfg_scale values (see `eval_llada.sh`)
- CFG implementation: `logits = un_logits + (cfg_scale + 1) * (logits - un_logits)`

### Gumbel Noise
- Uses float64 for Gumbel max sampling to maintain generation quality
- Temperature parameter controls randomness (0 = greedy)

## Visualization Tools

Located in `visualization/`:
1. `generate.py`: Generate sampling process records
2. `visualization_paper.py` / `visualization_zhihu.py`: Create HTML visualizations
3. `html_to_png.py`: Convert HTML to PNG for GIF creation

## Common Workflows

### Test a model on a new prompt
```bash
python generate.py  # Modify the prompts in main()
```

### Evaluate on a specific benchmark
```bash
# For likelihood-based (Base model):
accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',cfg=0.0,mc_num=1

# For generation-based (Base model):
accelerate launch eval_llada.py --tasks gsm8k --model llada_dist --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024
```

### Debug model outputs
- Check the generation process by adding print statements in `generate.py` lines 77-119
- For likelihood issues, inspect `get_log_likelihood.py` lines 66-76

## Development Notes

- When modifying generation code, pay attention to the remasking strategy (line 100-107 in `generate.py`)
- The `num_transfer_tokens` function (lines 22-40 in `generate.py`) precomputes how many tokens to unmask at each step
- For evaluation debugging, `is_check_greedy=False` in `eval_llada.py` significantly speeds up evaluation
- Multi-GPU evaluation is supported via `accelerate` - the model handles distributed evaluation automatically
