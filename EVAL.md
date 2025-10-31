# Evaluation
In this file, we provide the code for the evaluation of [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base),
[LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and [LLaDA 1.5](https://arxiv.org/abs/2505.19223).


## Benchmarks
For **LLaDA-8B-Base**, we employ conditional likelihood estimation and conditional generation for evaluation following the 
widely adopted evaluation process in LLMs. Please refer to Appendix B.6 of our [paper](https://arxiv.org/pdf/2502.09992)
for details.

| Evaluation Method of LLaDA-8B-Base | MMLU | BBH | ARC-C | Hellaswag | TruthfulQA | WinoGrande | PIQA | GSM8K | Math | GPQA | HumanEval | HumanEval-FIM | MBPP | CMMLU | C-Eval |
|:----------------------------------|:----:|:----:|:------:|:-----------:|:------------:|:------------:|:----:|:----:|:----:|:----:|:-----------:|:---------------:|:----:|:----:|:----:|
| **Evaluation Type** | ppl | gen | ppl | ppl | ppl | ppl | ppl | gen | gen | ppl | gen | gen | gen | ppl | ppl |

where ppl refers to conditional likelihood estimation and gens refer to conditional generation.

Both **LLaDA-8B-Instruct** and **LLaDA 1.5** are evaluated using only conditional generation.

## Open source testing tools
For LLaDA-8B-Base, LLaDA-8B-Instruct, and LLaDA-1.5, we initially conducted evaluations using our internal benchmark suite. 
Recently, we reproduced our results using two open-source evaluation frameworks, [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness)
and [OpenCompass](https://github.com/open-compass/opencompass).

| Model | ppl tasks | gen tasks |
|:------|:-----------|:-----------|
| **LLaDA-8B-Base** | lm-eval | lm-eval / OpenCompass |
| **LLaDA-8B-Instruct & LLaDA-1.5** | None | OpenCompass |


## Usage
### lm-eval
Please refer to `eval_llada_lm_eval.sh` for the required dependencies and execution commands.

For **the ppl tasks of LLaDA-8B-Base**, the evaluation results are as follows:

|                | ARC-C | Hellaswag | TruthfulQA | WinoGrande | GPQA | PIQA | MMLU | CMMLU | C-Eval |
|----------------|:------:|:----------:|:-----------:|:-----------:|:----:|:----:|:----:|:----:|:----:|
| **w/o CFG**    | 45.9  | 70.5       | 46.1        | **74.8**    | 25.2 | 73.6 | 65.9 | 69.9 | 70.5 |
| **w/ CFG**     | **47.9** | **72.5** | **46.4**    | **74.8**    | **26.1** | **74.4** |  –   | – | – |

In the Tab.1 of [LLaDA paper](https://arxiv.org/pdf/2502.09992), we only report results w/o CFG to ensure a fair comparison
with autoregressive models. 


For **the gen tasks of LLaDA-8B-Base**, the evaluation result are as follows:

| Settings | BBH | GSM8K | Math | HumanEval | MBPP |
|:------------------------------------|:----:|:----:|:----:|:----:|:----:|
| **gen_length = 1024, steps = 1024, block_length = 1024** | 49.7 | 70.3 | 31.4 | 35.4 | 40.0 |
| **gen_length = 512, steps = 512, block_length = 512**   | 50.4 | 70.8 | 30.9 | 32.9 | 39.2 |
| **gen_length = 256, steps = 256, block_length = 256**   | 45.0 | 70.0 | 30.3 | 32.9 | 40.2 |

In the Tab.1 of [LLaDA paper](https://arxiv.org/pdf/2502.09992), we report the results with `gen_length = 1024, steps = 1024, block_length = 1024` for simplicity. 
However, as shown above, the performance across all three settings is consistent.


### OpenCompass
Please refer to `eval_llada_opencompass.sh` for the required dependencies and execution commands.

In addition to lm-eval, we can also employ OpenCompass to evaluate **LLaDA-8B-Base**. For the `gen_length = 256, steps = 256, block_length = 256` 
setting, the results are as follows:

| Settings        | BBH  | GSM8K | Math | HumanEval | MBPP |
|:----------------|:----:|:-----:|:----:|:---------:|:----:|
| **lm-eval**     | 45.0 | 70.0 | 30.3 | 32.9 | 40.2 |
| **OpenCompass** | 47.3 | 71.9  | 30.7 |   34.1   | 38.8 |


For **LLaDA-8B-Instruct**, the evaluation results are as follows. It is worth noting that in the Tab.1 and Tab.2 of [LLaDA paper](https://arxiv.org/pdf/2502.09992),
we report the results with **pure diffusion sampling without any autoregressive elements**, as this setting yields the best overall performance.

|                        | MMLU | MMLU-pro | Hellaswag | ARC-C | GSM8K | Math  | GPQA | HumanEval | MBPP |
|:-----------------------|:----:|:--------:|:---------:|:-----:|:-----:|:-----:|:----:|:----------:|:----:|
| **gen\_length**        | 3    | 256      | 3         | 512   | 512   | 512   | 64   | 512        | 256  |
| **block\_length**      | 3    | 256      | 3         | 512   | 512   | 512   | 64   | 512        | 256  |
| **logits\_eos\_inf**   | False| False    | False     | False | False | False | False| True       | False|
| **confidence\_eos\_eot\_inf** | False| False| False | False | True  | True  | True | False      | True |
| **Internal toolkit**   | 65.5 | 37.0     | 74.6      | 88.5  | 69.4  | 31.9  | 33.3 | 49.4       | 41.0 |
| **OpenCompass**        | 65.4 | 36.6     | 75.3      | 89.2  |  68.8 |   29.6   | 32.3 | 47.0       | 39.6 |

Please refer to Appendix B.4 of [LLaDA paper](https://arxiv.org/pdf/2502.09992) for the explanation of the sampling setting.

Furthermore, we apply block diffusion sampling (i.e., semi-autoregressive remasking) to mitigate the tendency of **LLaDA-8B-Instruct**
to generate excessive |EOS| tokens, which is caused by the extensive |EOS| padding in the SFT data. This strategy improves performance 
on the GSM8K and Math benchmarks, while leading to a decrease in accuracy on other benchmarks.

|                        | GSM8k | Math  |
|:-----------------------|:-----:|:-----:|
| **gen\_length**        |  256  |  512  |
| **block\_length**      |   8   |  64   |
| **logits\_eos\_inf**   | False | False | 
| **confidence\_eos\_eot\_inf** | False | False | 
| **Internal toolkit**   | 78.6  | 42.2  | 
| **OpenCompass**        | 78.9  | 42.7  | 


The evaluation results of **LLaDA 1.5** are as follows:

|                           | GSM8K | Math | GPQA | HumanEval | MBPP | IFEval |
|:--------------------------|:-----:|:----:|:----:|:---------:|:----:|:------:|
| **gen_length**            |  256  | 1024 | 256  |    512    | 512  |  256   |
| **block_length**          |   16  | 128  |  16  |     32    |  32  |   16   |
| **logits_eos_inf**        | False | False| False|   False   | False| False  |
| **confidence_eos_eot_inf**| True  | True | False|   True    | True |  True  |
| **Internal toolkit**      | 83.8  | 42.6 | 36.9 |   52.4    | 42.8 |  66.2  |
| **OpenCompass**           | 83.6  | 42.3 | 34.8 |   51.2    | 42.6 |  65.2  |

Note that Arena-Hard, AlignBench, and MT-Bench require access to the OpenAI API for evaluation, and are therefore not included.

**Batch generation** is also supported in OpenCompass. To enable this feature, update the `batch_size` and `batch_size_` parameters in `OpenCompass/examples/xxx.py`.
Please note that in October 2025, we updated the `modeling_llada.py` file in Hugging Face to support attention mask inputs. 
Make sure you are using the latest version of the file to ensure compatibility. 

If you want to use a **custom model path**, edit the model file under `opencompass/opencompass/configs/models/dllm/xxx.py` and modify the path argument. 
For example:
```python
models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct',
        path='/your/custom/path/to/GSAI-ML/LLaDA-8B-Instruct',  # Change this path
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
```

## Reversal curse
We downloaded a [text file](https://wenku.baidu.com/view/f13866185fbfc77da369b1b3?wkts=1760409102730) containing a large collection of classical Chinese poetic lines from Baidu Wenku.
Using regular expressions, we extracted pairs of consecutive poetic lines (i.e., couplets) and stored them in a file named `data/poem_data.json`.

We provide the evaluation command as follows:
```
# generate the subsequent line
python eval_reverse.py  --type ftb --eos_inf

# generate the preceding line
python eval_reverse.py  --type btf --eos_inf
```

## Acknowledgments
Thanks [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [OpenCompass](https://github.com/open-compass/opencompass)
for their great work!












