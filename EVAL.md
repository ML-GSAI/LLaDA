# Eval

## Usage for PPL evaluation
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
## Usage for Gen evaluation
For gen evaluation, we use the open-source toolkit `opencompass` to reproduce our results. Please see [Opencompass](./opencompass/README.md) for more detials.

For LLaDA-Instruct, we provide a comparison of nine generation metrics evaluated using both the open-source `opencompass` library and our internal evaluation toolkit.
| Benchmark | Internal toolkit | `Opencompass` |
|-----------|-------------|-------------------------------------|
| mmlu      | 65.5     | 65.36                              |
| mmlu_pro  | 37       | 36.58                              |
| Hellaswag | 74.6     | 75.32                              |
| ARC-C     | 88.5     | 89.15                              |
| GSM8K     | 78.6     | 79.23                              |
| math      | 26.6     | 26.34                              |
| GPQA      | 31.8     | 30.30                             |
| HumanEval | 47.6     | 48.78                              |
| MBPP      | 34.2     | 37.6                               |

For LLaDA-Base, we set the number of generation steps to 256 and capped the response length at 256 tokens.
| Benchmark | Internal toolkit | `Opencompass` |
|-----------|-------------|-------------------------------------|
| GSM8K     | 70.7     | 72.78                              |
| math      | 27.3     | 30.02                              |
| bbh      | 49.8     |  47.51                          |
| HumanEval | 33.5     | 32.92                              |
| MBPP      | 38.2     | 39.6                              |

For LLaDA-1.5, we follow the experimental setup detailed in [LLaDA-1.5](https://arxiv.org/abs/2505.19223). For IFEval Bench, we report the average of `Prompt-level-strict-accuracy` and `Inst-level-strict-accuracy`.
| LLaDA-instruct | Internal toolkit | `Opencompass` |
|-----------|-------------|-------------------------------------|
| GSM8K     | 78.6     | 78.85                             |
| math      | 42.2   | 43.2                              |
| GPQA      | 33.3     | 32.32                              |
| HumanEval | 49.4     | 46.95                             |
| MBPP      | 41.0     | 39.6                               |
| IFEval      | 62.2     | 64.41                               |


| LLaDA-1.5 | Internal toolkit | `Opencompass` |
|-----------|-------------|-------------------------------------|
| GSM8K     | 82.8     | 83.62                              |
| math      | 42.3     | 42.34                             |
| GPQA      | 36.4     | 34.85                              |
| HumanEval | 51.2     | 51.22                             |
| MBPP      | 42.8    | 42.6                               |
| IFEval      | 66.2     | 65.24                               |


We were unable to reproduce the **same** results reported in the paper due to the following key factors:
1. *Prompt Discrepancies*: The internal toolkit utilizes different prompts, which led to variations in the results.
2. *Divergent Evaluation Settings*: The use of internal dataset code resulted in a different evaluation setup. For example, our version of the GPQA dataset differs from that used in the open-source OpenCompass framework.
3. *Hardware-related Precision Errors*: Computational precision differences between H100 and A100 series GPUs introduced a result variance of over 2%.

Nevertheless, we successfully replicated our paper's findings using the OpenCompass framework. The remaining benchmark evaluations will be completed and reported subsequently.

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
