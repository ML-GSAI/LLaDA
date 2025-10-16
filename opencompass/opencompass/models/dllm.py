import os
import sys
from pathlib import Path
llada_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(llada_root))
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
##use llada generate
from generate import generate as LLaDA_generate
import torch.nn.functional as F
import numpy as np
PromptType = Union[PromptList, str]
def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            # XXX: all system roles are mapped to human in purpose
            # dict(role='SYSTEM', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')]
    )
    return APITemplateParser(meta_template or default_meta_template)

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@MODELS.register_module()
class LLaDAModel(BaseModel):
    """Model wrapper around LLaDA model.

    Args:
        path (str): The name or path to LLaDA model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
        pad_token_id (int): The id of the padding token. Defaults to None. Use
            (#vocab + pad_token_id) if get negative value.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        use_fastchat_template (str, optional): Whether to use fastchat to get
            the conversation template. If True, fastchat needs to be
            implemented first. Defaults to False.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None,
                 stop_words: Optional[List[str]] = [],
                 cfg = 0,
                 temperature = 0.,
                 remasking = 'low_confidence', # 'random'
                 mask_id = 126336, # The token id of [MASK] is 126336.
                 padding_id = 126081, # The token id of <pad> is 126081.
                 mc_num = 1,
                 gen_steps = 512,
                 gen_length = 512,
                 gen_blocksize = 512,
                 batch_size_ = 1,
                 diff_confidence_eos_eot_inf = False,
                 diff_logits_eos_inf = False,
                 ) -> None:
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str
        self.stop_words = stop_words
        ## add cfg and mc_num
        self.cfg = cfg
        self.mc_num = mc_num

        ##Todo: modify input
        self.batch_size = batch_size_
        self.gen_steps = gen_steps
        self.gen_length = gen_length
        self.gen_blocksize = gen_blocksize
        self.temperature = temperature
        self.remasking = remasking
        self.padding_id = padding_id
        self.mask_id = mask_id
        self.diff_confidence_eos_eot_inf = diff_confidence_eos_eot_inf
        self.diff_logits_eos_inf = diff_logits_eos_inf

        self.template_parser = _get_meta_template(meta_template)

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    'pad_token_id is not consistent with the tokenizer. Using '
                    f'{self.pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None
            }.get(model_kwargs['torch_dtype'])
        self.logger.debug(f'HF using torch_dtype: {torch_dtype}')
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code = True,**model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id


    def _get_loglikelihood(self, inputs: str, conts: str) -> float:
        """Get loglikelihood scores given input string and continuation string.

        Args:
            inputs (str): string.
            conts (str): strings: slices after the space.
        Returns:
            float: loglikelihood scores.
        """
        input_tokenizer_out = self.tokenizer(inputs,
                                             padding=True,
                                             truncation=False,
                                             return_length=True,
                                             return_tensors='pt').to(
                                                 self.model.device)

        input_ids = input_tokenizer_out['input_ids'][:, :self.max_seq_len]
        input_length = input_tokenizer_out['length']
        context_ids = [
            self.tokenizer(inputs[i].replace(conts[i], ''),
                           padding=False,
                           truncation=True,
                           max_length=self.max_seq_len)['input_ids']
            for i in range(len(inputs))
        ]
        # forward
        outputs = self.model(input_ids)['logits']
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # calculate loglikelihood
        answer = np.zeros(len(inputs))
        for i in range(len(inputs)):
            if self.tokenizer.padding_side == 'right':
                cont_ids = input_ids[i, len(context_ids[i]):input_length[i]]
                logits = outputs[i,
                                 len(context_ids[i]) - 1:input_length[i] -
                                 1, :]  # noqa
            else:
                cont_ids = input_ids[i, len(context_ids[i]) - input_length[i]:]
                logits = outputs[i,
                                 len(context_ids[i]) - input_length[i] - 1:-1]
            # Reducing the dimension will lead to a wrong outcome
            logits_gather = torch.gather(
                logits.unsqueeze(0), 2,
                cont_ids.unsqueeze(0).unsqueeze(-1))  # [1, seq]
            # Answer: sum the likelihood of each token in continuation
            answer[i] = float(logits_gather.detach().cpu().sum())
        return answer

    def get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_mink_percent(inputs, k=k)
        else:
            return np.concatenate([
                self._get_mink_percent(inputs=[text], k=k) for text in inputs
            ])

    def _get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = inputs['tokens']['input_ids'][:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = -nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        messages = _convert_chat_messages(inputs)
        prompt = [self.tokenizer.apply_chat_template(m_i, add_generation_prompt=True, tokenize=False) for m_i in messages]
        print('steps:', self.gen_steps, 'length:', self.gen_length, 'blocksize:', self.gen_blocksize)
        print('temperature:', self.temperature, 'cfg:', self.cfg, 'remasking:', self.remasking)
        print('mask_id:', self.mask_id, 'padding_id:', self.padding_id)
        print('diff_confidence_eos_eot_inf:', self.diff_confidence_eos_eot_inf, 'diff_logits_eos_inf:', self.diff_logits_eos_inf)
        print('final prompt:', prompt)
        prompt = self.tokenizer.batch_encode_plus(prompt, padding = True, return_tensors='pt')['input_ids']
        responses = LLaDA_generate(
            model = self.model,
            prompt = prompt.to(self.model.device),
            tokenizer = self.tokenizer,
            steps = self.gen_steps,
            gen_length = self.gen_length,
            block_length = self.gen_blocksize,
            temperature = self.temperature,
            cfg_scale = self.cfg,
            remasking = self.remasking,
            mask_id = self.mask_id,
            padding_id = self.padding_id,
            diff_confidence_eos_eot_inf = self.diff_confidence_eos_eot_inf,
            diff_logits_eos_inf = self.diff_logits_eos_inf,
        )
        return responses
    
    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        raise NotImplementedError('Please use `lmeval` toolkit instead.')

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            if mask_length is not None:
                print('_______')
                return np.concatenate([
                    self._get_ppl(inputs=[text],
                                         mask_length=[mask_length[idx]])
                    for idx, text in enumerate(inputs)
                ])
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def get_loglikelihood(self, inputs: List[str], conts:  List[str]) -> List[float]:
        print('inputs:', inputs, 'conts:', conts)
        mask_length = [self.get_token_len(c, add_special_tokens=False) for c in conts]
        return - self.get_ppl(inputs, mask_length)
    
    def get_logits(self, inputs, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == inputs.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(inputs.shape[0], 1)
            un_inputs = inputs.clone()
            un_inputs[prompt_index] = self.mask_id
            inputs = torch.cat([inputs, un_inputs])

        logits = self.model(inputs).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :inputs.shape[1]]

def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs

class LLaDABaseModel(LLaDAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.template_parser = LMTemplateParser()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        messages = _convert_base_messages(inputs)
        prompt = messages
        print('steps:', self.gen_steps, 'length:', self.gen_length, 'blocksize:', self.gen_blocksize)
        print('temperature:', self.temperature, 'cfg:', self.cfg, 'remasking:', self.remasking)
        print('mask_id:', self.mask_id, 'padding_id:', self.padding_id)
        print('diff_confidence_eos_eot_inf:', self.diff_confidence_eos_eot_inf, 'diff_logits_eos_inf:', self.diff_logits_eos_inf)
        print('final prompt:', prompt)
        prompt = self.tokenizer.batch_encode_plus(prompt, padding = True, return_tensors='pt')['input_ids']
        responses = LLaDA_generate(
            model = self.model,
            prompt = prompt.to(self.model.device),
            tokenizer = self.tokenizer,
            steps = self.gen_steps,
            gen_length = self.gen_length,
            block_length = self.gen_blocksize,
            temperature = self.temperature,
            cfg_scale = self.cfg,
            remasking = self.remasking,
            mask_id = self.mask_id,
            padding_id = self.padding_id,
            diff_confidence_eos_eot_inf = self.diff_confidence_eos_eot_inf,
            diff_logits_eos_inf = self.diff_logits_eos_inf,
        )
        batch_size = prompt.shape[0]
        stopping_criteria = set(self.stop_words)
        for i in range(batch_size):
            response = responses[i]
            for stop_word in stopping_criteria:
                if stop_word in response:
                    response = response.split(stop_word)[0]
                    break
            responses[i] = response
        return responses
    
    def get_token_len(self, prompt: str, add_special_tokens: bool=True) -> int:
        m = _convert_base_messages([prompt])[0]
        t = self.tokenizer(m, add_special_tokens=add_special_tokens)
        return len(t['input_ids'])
    