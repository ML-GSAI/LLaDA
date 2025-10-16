# llada arxiv paper setting for llada instruct model
python run.py examples/llada_instruct_gen_gsm8k.py -w outputs/llada_instruct_gsm8k
python run.py examples/llada_instruct_gen_math.py -w outputs/llada_instruct_math
python run.py examples/llada_instruct_gen_mmlu.py -w outputs/llada_instruct_mmlu
python run.py examples/llada_instruct_gen_mmlupro.py -w outputs/llada_instruct_mmlupro
python run.py examples/llada_instruct_gen_hellaswag.py -w outputs/llada_instruct_hellaswag
python run.py examples/llada_instruct_gen_arcc.py -w outputs/llada_instruct_arcc
python run.py examples/llada_instruct_gen_gpqa.py -w outputs/llada_instruct_gpqa
python run.py examples/llada_instruct_gen_humaneval.py -w outputs/llada_instruct_humaneval
python run.py examples/llada_instruct_gen_mbpp.py -w outputs/llada_instruct_mbpp

# llada arxiv paper setting for llada base model
python run.py examples/llada_base_gen_gsm8k.py -w outputs/llada_base_gsm8k
python run.py examples/llada_base_gen_math.py -w outputs/llada_base_math
python run.py examples/llada_base_gen_humaneval.py -w outputs/llada_base_humaneval
python run.py examples/llada_base_gen_mbpp.py -w outputs/llada_base_mbpp
python run.py examples/llada_base_gen_bbh.py -w outputs/llada_base_bbh

# llada 1.5 paper setting for llada instruct model
python run.py examples/llada_gen_gsm8k_15.py -w outputs/llada_instruct_gsm8k_15
python run.py examples/llada_gen_math_15.py -w outputs/llada_instruct_math_15
python run.py examples/llada_gen_gpqa_15.py -w outputs/llada_instruct_gpqa_15
python run.py examples/llada_gen_humaneval_15.py -w outputs/llada_instruct_humaneval_15
python run.py examples/llada_gen_mbpp_15.py -w outputs/llada_instruct_mbpp_15
python run.py examples/llada_gen_ifeval_15.py -w outputs/llada_instruct_ifeval_15

# llada 1.5 paper setting for llada 1.5 model
python run.py examples/llada_15_gen_gsm8k.py -w outputs/llada_15_gsm8k
python run.py examples/llada_15_gen_math.py -w outputs/llada_15_math
python run.py examples/llada_15_gen_gpqa.py -w outputs/llada_15_gpqa
python run.py examples/llada_15_gen_humaneval.py -w outputs/llada_15_humaneval
python run.py examples/llada_15_gen_mbpp.py -w outputs/llada_15_mbpp
python run.py examples/llada_15_gen_ifeval.py -w outputs/llada_15_ifeval
