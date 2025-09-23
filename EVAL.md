# Eval

## Usage
Please refer to `eval_llada.sh` for the required dependencies and execution commands.

For LLaDA-Base, we provide a comparison of the five conditional generation metrics evaluated using both the open-source `lm-eval` library and our internal evaluation toolkit.

||BBH|GSM8K|Math|HumanEval|MBPP|
|-|-|-|-|-|-|
|Internal toolkit|49.8|70.7|27.3|33.5|38.4|
|`lm-eval`|49.7|70.3|31.4|35.4|40.0|

In addition, we provide ablation studies on the above five metrics with respect to different generation lengths using `lm-eval`.
||BBH|GSM8K|Math|HumanEval|MBPP|
|-|-|-|-|-|-|
|gen_length=1024,steps=1024,block_length=1024|49.7|70.3|31.4|35.4|40.0|
|gen_length=512,steps=512,block_length=512|50.4|70.8|30.9|32.9|39.2|
|gen_length=256,steps=256,block_length=256|45.0|70.0|30.3|32.9|40.2|

For LLaDA-Instruct, we provide a comparison of nine generation metrics evaluated using both the open-source `opencompass` library and our internal evaluation toolkit. For the MBPP code generation benchmark, we employ a 3-shot prompting configuration.
| Benchmark | Internal toolkit | `Opencompass` |
|-----------|-------------|-------------------------------------|
| mmlu      | 65.5     | 64.89                              |
| mmlu_pro  | 37       | 36.58                              |
| Hellaswag | 74.6     | 75.32                              |
| ARC-C     | 88.5     | 89.15                              |
| GSM8K     | 78.6     | 79.60                              |
| math      | 26.6     | 26.34                              |
| GPQA      | 31.8     | 32.32                              |
| HumanEval | 47.6     | 48.78                              |
| MBPP      | 34.2(4)     | 38.2(3)                               |


## Challenges encountered when reproducing the Instruct model with `lm-eval`
To ensure that we was using `lm-eval` correctly, we first tested it on **LLaMA3-8B-Instruct**. The results are as follows:

||MMLU|MMLU Pro|ARC-C|GSM8K|Math|GPQA|HumanEval|MBPP|
|-|-|-|-|-|-|-|-|-|
|[Reported](https://arxiv.org/pdf/2407.10671)|68.4|41.0|-|79.6|30.0|34.2|62.2|67.9|
|Internal toolkit|68.4|41.9|82.4|78.3|29.6|33.5|59.8|57.6|
|`lm-eval`|66.5|19.6|82.1|67.3|27.3|33.5|36.6|57.0|

We found that for benchmarks such as MMLU-Pro, GSM8K, and HumanEval, the results obtained using `lm-eval` are significantly lower than expected. Once we resolve the issues affecting the evaluation of LLaMA3-8B-Instruct, we will release the evaluation code for LLaDA-Instruct.

If you have any suggestions or feedback on this BUG, please feel free to contact us via email at nieshen@ruc.edu.cn or reach out via WhatsApp/WeChat at (+86) 18809295303. We would greatly appreciate it.

Below is the command we used to test LLaMA3-8B-Instruct:
```
pip install transformers==4.49.0 accelerate==0.34.2
pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .


export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks mmlu_generative,gpqa_main_generative_n_shot,gsm8k \
    --num_fewshot 5 \
    --trust_remote_code \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto:4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks minerva_math \
    --num_fewshot 4 \
    --trust_remote_code \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto:4

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks mmlu_pro,arc_challenge_chat \
    --trust_remote_code \
    --apply_chat_template \
    --batch_size auto:4

# For HumanEval and MBPP, using --apply_chat_template leads to significantly lower final results.
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks humaneval_instruct,mbpp \
    --trust_remote_code \
    --confirm_run_unsafe_code \
    --batch_size auto:4

```
