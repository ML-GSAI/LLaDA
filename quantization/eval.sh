MODEL_PATH=/home/chentianqi/deepseek/GPTQModel/models/LLaDA-8B-Instruc-default:w4g128

#Instruct model 
CUDA_VISIBLE_DEVICES=0 python  eval_llada.py --tasks gsm8k --num_fewshot 4 --model llada_dist \
    --apply_chat_template \
    --batch_size 1 --model_args "model_path=$MODEL_PATH,cfg=0.0,is_check_greedy=False,max_length=256,block_length=8,steps=256"

#Base model
accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path=$MODEL_PATH,cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path=$MODEL_PATH,cfg=2.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path=$MODEL_PATH,cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path=$MODEL_PATH,cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --model llada_dist --batch_size 8 --model_args model_path=$MODEL_PATH,cfg=0.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --model llada_dist --batch_size 8 --model_args model_path=$MODEL_PATH,cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path=$MODEL_PATH,cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path=$MODEL_PATH,cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --model llada_dist --batch_size 1 --model_args model_path=$MODEL_PATH,cfg=0.0,is_check_greedy=False,mc_num=1
