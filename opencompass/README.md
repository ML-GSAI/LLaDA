# LLaDA-Instruct Evaluation Toolkit
This toolkit contains the evaluation code for LLaDA-Instruct and LLaMA-Instruct models.

## Quickstart

### Installation
To install the toolkit, follow the instructions from [open-compass/opencompass](https://github.com/open-compass/opencompass):
```bash
conda create --name llada_eval python=3.10 -y
conda activate llada_eval
pip install -e .
```

We also provide scripts to evaluate [GSAI-ML/LLaDA-8B-Instruct](https://hf-mirror.com/GSAI-ML/LLaDA-8B-Instruct/tree/main) , [GSAI-ML/LLaDA-1.5](https://hf-mirror.com/GSAI-ML/LLaDA-1.5/tree/main) and [meta-llama/Meta-Llama-3-8B-Instruct](https://hf-mirror.com/meta-llama/Meta-Llama-3-8B-Instruct):
```
bash eval_llada_instruct.sh
bash eval_llama_instruct.sh
```

### HumanEval Setup
For HumanEval evaluation, install the additional dependency:
```bash
git clone https://github.com/open-compass/human-eval.git
cd human-eval && pip install -e .
```

### MATH Setup
For Math evaluation, pip install the additional dependency:
```bash
pip install math_verify latex2sympy2_extended
```
### Custom Model Path
If you want to use a custom model path, edit the model file under `opencompass/opencompass/configs/models/dllm/` and modify the path argument. For example:
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

### Batch Generate
We also implement the `batch_generate` function. To use this fuction, please set `batch_size >1, batch_size_>1, batch_size = batch_size_`. You should also use the new [modeling_llada](../modeling_llada.py) instead of the default one.

## Acknowledgement
This project is a fork of [open-compass/opencompass](https://github.com/open-compass/opencompass), an open-source large model evaluation platform. The original OpenCompass framework provides the foundation for our LLaDA-Instruct and LLaMA-Instruct model evaluations.