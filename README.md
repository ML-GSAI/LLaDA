# Large Language Diffusion Models
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2502.09992)
[![deploy](https://img.shields.io/badge/Hugging%20Face-LLaDA_Base-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
[![deploy](https://img.shields.io/badge/Hugging%20Face-LLaDA_Instruct-FFEB3B)](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
[![deploy](https://img.shields.io/badge/Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/multimodalart/LLaDA)
[![deploy](https://img.shields.io/badge/Zhihu1-知乎1-blue)](https://zhuanlan.zhihu.com/p/24214732238)
[![deploy](https://img.shields.io/badge/Zhihu2-知乎2-blue)](https://www.zhihu.com/question/1908479621466396378/answer/1910672718174589774?share_code=1kreOq5gzOtnM&utm_psn=1910708245535912148&utm_source=wechat_timeline&utm_medium=social&s_r=0)

## News
### New works
- [2025.02.14] We have uploaded LLaDA paper to [arXiv](https://arxiv.org/abs/2502.09992) and open-sourced [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).

- [2025.05.23] We introduce [LLaDA-V](https://ml-gsai.github.io/LLaDA-V-demo/), a competitive diffusion-based vision-language model, outperforming other diffusion MLLMs.

- [2025.05.25] We introduce [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/), which incorporates VRPO to reduce gradient variance and enhance preference alignment in LLaDA.

- [2025.09.11] We introduce [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) and [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct), the first diffusion language model pretrained from scratch with MoE architecture. LLaDA-MoE-7B-A1B-Instruct uses only ~1B active parameters at inference while surpassing LLaDA 1.5(an 8B dense model), and comparable to Qwen2.5-3B-Instruct.


### New features in this repo
- [2025.05.04] We have provided evaluation code based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the LLaDA-8B-Base.

- [2025.10.27] We have provided batch inference support, along with all evaluation code for [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base), [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and [LLaDA 1.5](https://ml-gsai.github.io/LLaDA-1.5-Demo/).

  
## Introduction
We introduce LLaDA (<b>L</b>arge <b>La</b>nguage <b>D</b>iffusion with m<b>A</b>sking), a diffusion model with an unprecedented 8B scale, trained entirely from scratch, 
rivaling LLaMA3 8B in performance.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/LLaDA_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/LLaDA_vs_LLaMA_chat.svg" style="width: 46%" />
</div>


## Inference
The [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) are uploaded
in Huggingface. Please first install `transformers==4.38.2` and employ the [transformers](https://huggingface.co/docs/transformers/index) to load.

```angular2html
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

We provide `get_log_likelihood()` and `generate()` functions in `get_log_likelihood.py` 
and `generate.py` respectively, for conditional likelihood evaluation and conditional generation.

You can directly run `python chat.py` to have multi-round conversations with LLaDA-8B-Instruct.


## Gradio demo 
Thank you very much to [apolinário](https://github.com/apolinario) for helping us create this amazing demo!

First, install [Gradio](https://www.gradio.app) `pip install gradio`, and then you can directly run `python app.py`

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Pre-training and Supervised Fine-Tuning

We will not provide the training framework and data as most open-source LLMs do.

However, the pre-training and Supervised Fine-Tuning of LLaDA are straightforward. If 
you have a codebase for training an autoregressive model, you can modify it to 
adapt to LLaDA with just a few lines of code.

We provide guidelines for the pre-training and SFT of LLaDA in [GUIDELINES.md](GUIDELINES.md). 
You can also refer to [SMDM](https://github.com/ML-GSAI/SMDM), which has a similar training process to LLaDA 
and has open-sourced the training framework.

## Evaluation
Please refer to [EVAL.md](EVAL.md) for instructions on using the evaluation code.

## FAQ
Here, we address some common questions about LLaDA.

### 0. How do I train my own LLaDA?
Please refer to [GUIDELINES.md](GUIDELINES.md) for the guidelines. 
You can also refer to [SMDM](https://github.com/ML-GSAI/SMDM), which follows the same training 
process as LLaDA and has open-sourced its code.


### 1. What is the difference between LLaDA and BERT?

Our motivation is not to improve BERT, nor to apply image generation methods like [MaskGIT](https://arxiv.org/abs/2202.04200) 
to text. **Our goal is to explore a theoretically complete language modeling approach — masked diffusion models.** 
During this process, we simplified the approach and discovered that the loss function of masked diffusion models 
is related to the loss functions of BERT and MaskGIT. You can find our theoretical research process in Question 7.

Specifically, LLaDA employs a masking ratio that varies randomly between 0 and 1, while BERT uses 
a fixed ratio. This subtle difference has significant implications. **The training
objective of LLaDA is an upper bound on the negative log-likelihood of the model 
distribution, making LLaDA a generative model.** This enables LLaDA to naturally 
perform in-context learning, instruction-following, and ensures Fisher consistency 
for scalability with large datasets and models. You can also find a direct answer 
to this question in Section 2.1 of our paper.


### 2. What is the relationship between LLaDA and Transformer?
Network structure and probabilistic modeling are two distinct approaches that collectively form the 
foundation of language models. LLaDA, like GPT, adopts the 
Transformer architecture. The key difference lies in the probabilistic modeling approach: GPT 
utilizes an autoregressive next-token prediction method, 
while LLaDA employs a diffusion model for probabilistic modeling.


### 3. What is the sampling efficiency of LLaDA?
Currently, LLaDA's sampling speed is slower than the autoregressive baseline for three reasons: 
1. LLaDA samples with a fixed context length;
2. LLaDA cannot yet leverage techniques like KV-Cache;
3. LLaDA achieves optimal performance when the number of sampling steps equals the response length.
Reducing the number of sampling steps leads to a decrease in performance, as detailed in Appendix B.4 
and Appendix B.6 of our paper.

In this work, we aim to explore the upper limits of LLaDA's capabilities, **challenging the assumption 
that the key LLM abilities are inherently tied to autoregressive models**. We will continue 
to optimize its efficiency in the future. We believe this research approach is reasonable, 
as verifying the upper limits of diffusion language models' capabilities will provide us with
more resources and sufficient motivation to optimize efficiency.

Recall the development of diffusion models for images, from [DDPM](https://arxiv.org/abs/2006.11239) 
to the [Consistency model](https://arxiv.org/pdf/2410.11081), where sampling speed accelerated nearly 
1000 times over the course of 4 years. **We believe there is significant room for optimization in LLaDA's 
sampling efficiency as well**. Current solutions, including [block diffusion](https://arxiv.org/abs/2503.09573), can mitigate the fixed context length issue, and 
[consistency distillation](https://arxiv.org/pdf/2502.05415) can reduce the number of sampling steps. In
addition, some cache methods (e.g., [Fast-dllm](https://github.com/NVlabs/Fast-dLLM), [dllm-cache](https://github.com/maomaocun/dLLM-cache))
can also be adapted by LLaDA.


### 4. What is the training stability of LLaDA?
For details on the pre-training process of LLaDA, please refer to Section 2.2 of our paper. 
During the total pre-training on 2.3T tokens, we encountered a training crash (loss becoming NaN) 
only once at 1.2T tokens. Our solution was to resume the checkpoint and reduce 
the learning rate from 4e-4 to 1e-4.


### 5. Why is the final answer "72" generated earlier than the intermediate calculation step (e.g., 12 × 4 = 48) in Tab4?

**The mask predictor has successfully predicted the reasoning process. However, during the 
remasking process, the reasoning steps are masked out again.** As shown in the figure 
below, the non-white background represents the model's generation process, while the 
white-background boxes indicate the predictions made by the mask predictor at each step. 
We adopt a randomly remasking strategy.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/diff_remask.gif" style="width: 80%" />
</div>

### 6. Why does LLaDA answer 'Bailing' when asked 'Who are you'?
This is because our pre-training and SFT data were designed for training an autoregressive model, 
whereas LLaDA directly utilizes data that contains identity markers.


### 7. Our journey in developing LLaDA?
LLaDA is built upon our two prior works, [RADD](https://arxiv.org/abs/2406.03736) and 
[SMDM](https://arxiv.org/abs/2410.18514). 

RADD demonstrated that the **training objective of LLaDA serves as an upper bound on the negative 
log-likelihood** of the model’s distribution, a conclusion also supported by [MD4](https://arxiv.org/abs/2406.04329) 
and [MDLM](https://arxiv.org/abs/2406.07524). 
Furthermore, RADD was the first to theoretically prove that **masked diffusion models do not require time t 
as an input to Transformer**. This insight provides the theoretical 
justification for LLaDA’s unmodified use of the Transformer architecture. Lastly, 
RADD showed that **the training objective of masked diffusion models is equivalent to that of 
any-order autoregressive models**, offering valuable insights into how masked diffusion models can 
overcome the reversal curse.

SMDM introduces the first **scaling law** for masked diffusion models and demonstrates that, with the 
same model size and training data, masked diffusion models can achieve downstream benchmark results 
on par with those of autoregressive models. Additionally, SMDM presents a simple, **unsupervised 
classifier-free guidance** method that greatly improves downstream benchmark performance, which has 
been adopted by LLaDA.


## Citation

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```
