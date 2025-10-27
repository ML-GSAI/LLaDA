import torch
import json
import argparse
import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from generate import generate


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gen_length', type=int, default=28, help='generation length')
    parse.add_argument('--block_length', type=int, default=28, help='block length')
    parse.add_argument('--cfg', type=float, default=0., help='classfier-free guidance scale')
    parse.add_argument('--eos_inf', action='store_true', help='set eos token logit to -inf')
    parse.add_argument('--type', type=str, default='ftb', help='btf (backward to forward): predict previous sentence, ftb (forward to backward): predict next sentence')
    args = parse.parse_args()
    return args
args = parse_args()


# Engish translation, extra_prompt = ' just output the sentence directly.'
extra_prompt = ' 直接输出句子即可。'
with open("data/poem_data.json", "r") as f:
    poems = json.load(f)


def next_predition_pairs(poems):
    return [poem['first'] + "的下一句是什么？" + extra_prompt for poem in poems], [poem['second'] for poem in poems]


def prev_predition_pairs(poems):
    return [poem['second'] + "的上一句是什么？" + extra_prompt for poem in poems], [poem['first'] for poem in poems]


device = 'cuda'
model = AutoModelForCausalLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

if args.type == 'ftb':
    prompts, answers = next_predition_pairs(poems)
elif args.type == 'btf':
    prompts, answers = prev_predition_pairs(poems)
else:
    raise NotImplementedError(args.type)


acc = 0
for index in tqdm.tqdm(range(len(prompts))):
    prompt, answer = prompts[index], answers[index]

    m = [{"role": "user", "content": prompt},]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=args.gen_length, temperature=0., cfg_scale=args.cfg,
                      gen_length=args.gen_length, block_length=args.block_length, logits_eos_inf=args.eos_inf)

    out = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    acc = acc + 1 if answer in out else acc

print(args)
print(f'Accuracy: {acc/ len(prompts)}')
print('*' * 20)
