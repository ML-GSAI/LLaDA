import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
import torch.nn.functional as F
import numpy as np
PromptType = Union[PromptList, str]
# @MODELS.register_module()
# class LLaDAModel(BaseModel):
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
#     def __init__(self,
#                  pkg_root: str,
#                  ckpt_path: str,
#                  tokenizer_only: bool = False,
#                  meta_template: Optional[Dict] = None,
#                  **kwargs):
#         super().__init__(pkg_root, ckpt_path, tokenizer_only, meta_template, **kwargs)

#     def add_gumbel_noise(logits, temperature):
#         '''
#         The Gumbel max is a method for sampling categorical distributions.
#         According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
#         Thus, we use float64.
#         '''
#         if temperature == 0:
#             return logits
#         logits = logits.to(torch.float64)
#         noise = torch.rand_like(logits, dtype=torch.float64)
#         gumbel_noise = (- torch.log(noise)) ** temperature
#         return logits.exp() / gumbel_noise


#     def get_num_transfer_tokens(mask_index, steps):
#         '''
#         In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#         Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#         the expected number of tokens transitioned at each step should be consistent.

#         This function is designed to precompute the number of tokens that need to be transitioned at each step.
#         '''
#         mask_num = mask_index.sum(dim=1, keepdim=True)

#         base = mask_num // steps
#         remainder = mask_num % steps

#         num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#         for i in range(mask_num.size(0)):
#             num_transfer_tokens[i, :remainder[i]] += 1

#         return num_transfer_tokens


#     @ torch.no_grad()
#     def _generate(self, prompt, steps=1024, gen_length=1024, block_length=1024, temperature=0.,
#                 cfg_scale=0., remasking='low_confidence', mask_id=126336):
#         '''
#         Args:
#             model: Mask predictor.
#             prompt: A tensor of shape (1, L).
#             steps: Sampling steps, less than or equal to gen_length.
#             gen_length: Generated answer length.
#             block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#             temperature: Categorical distribution sampling temperature.
#             cfg_scale: Unsupervised classifier-free guidance scale.
#             remasking: Remasking strategy. 'low_confidence' or 'random'.
#             mask_id: The toke id of [MASK] is 126336.
#         '''
#         x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#         x[:, :prompt.shape[1]] = prompt.clone()

#         prompt_index = (x != mask_id)

#         assert gen_length % block_length == 0
#         num_blocks = gen_length // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks

#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#             for i in range(steps):
#                 mask_index = (x == mask_id)
#                 if cfg_scale > 0.:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self.model(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self.model(x).logits

#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits, dim=-1)
#                     x0_p = torch.squeeze(
#                         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                     transfer_index[j, select_index] = True
#                 x[transfer_index] = x0[transfer_index]

#         return x
#     def get_token_len(self, prompt: str) -> int:
#         """Get lengths of the tokenized strings."""
#         return len(self.tokenizer.encode(prompt))

#     def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
#         """Generate results given a list of inputs. """
#         return self._generate(inputs, max_out_len)

#     def get_ppl(self,
#                 inputs: List[str],
#                 mask_length: Optional[List[int]] = None) -> List[float]:
#         """Get perplexity scores given a list of inputs."""
#         pass

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

    def batch_generate(self, prompt):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (B, L), potentially left-padded.
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The token id of [MASK].
            padding_id: The token id for padding.
        '''
        steps = self.gen_steps
        gen_length = self.gen_length
        block_length = self.gen_blocksize
        cfg_scale = self.cfg
        temperature = self.temperature
        remasking = self.remasking
        mask_id = self.mask_id
        padding_id = self.padding_id
        print('parameters:', steps, gen_length, block_length, temperature, cfg_scale, remasking, mask_id)
        messages = _convert_chat_messages(prompt)
        prompt = [self.tokenizer.apply_chat_template(m_i, add_generation_prompt=True, tokenize=False) for m_i in messages]
        print('final prompt:', prompt)
        prompt = self.tokenizer.batch_encode_plus(prompt, padding = True, return_tensors='pt')['input_ids']
        # prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids']
        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + gen_length
        
        is_left = False
        for i in range(batch_size):
            if (prompt[i, 0] == padding_id) and (prompt[i, -1] != padding_id):
                is_left = True
                break
        if not is_left:
            batch_padding_position = []
            for i in range(batch_size):
                padding_pos = (prompt[i] == padding_id).nonzero(as_tuple=True)[0]
                if len(padding_pos) > 0:
                    batch_padding_position.append(padding_pos[0].item())
                else:
                    batch_padding_position.append(prompt_len)
            
            ## where padding start
            prompt_len = batch_padding_position.copy()

        x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=self.model.device)
        if not is_left:
            for i in range(batch_size):
                x[i, :prompt_len[i]] = prompt[i, :prompt_len[i]].clone()
                x[i, prompt_len[i] + gen_length:] = torch.full((total_len - (prompt_len[i] + gen_length),), padding_id, dtype=torch.long) 
        else:
            x[:, :prompt_len] = prompt.clone()

        original_prompt_mask = (prompt != padding_id)
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        if not is_left:
            for i in range(batch_size):
                prompt_index[i, :prompt_len[i]] = original_prompt_mask[i, :prompt_len[i]]
        else:
            prompt_index[:, :prompt_len] = original_prompt_mask

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        
        attention_mask = torch.ones((batch_size, total_len), device=x.device, dtype=torch.long)
        
        if not is_left:
            for i in range(batch_size):
                # attention_mask[i, :prompt_len[i]+gen_length] = original_prompt_mask[i, :prompt_len[i]+gen_length].long()
                attention_mask[i, prompt_len[i] + gen_length:] = 0 
        else:
            attention_mask[:, :prompt_len] = original_prompt_mask.long()
        for num_block in range(num_blocks):

            if not is_left:
                start_pos = [prompt_len[i] + num_block * block_length for i in range(batch_size)]
                end_pos = [prompt_len[i] + (num_block + 1) * block_length for i in range(batch_size)]
                block_mask_index = torch.zeros((batch_size, block_length), dtype=torch.bool, device=x.device)
                for i in range(batch_size):
                    block_mask_index[i] = (x[i, start_pos[i]:end_pos[i]] == mask_id)
            else:
                start_pos = prompt_len + num_block * block_length
                end_pos = prompt_len + (num_block + 1) * block_length
                block_mask_index = (x[:, start_pos:end_pos] == mask_id)
            
           
            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                if not mask_index.any():
                    break
                    
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    
                    x_ = torch.cat([x, un_x], dim=0)
                    extended_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                    
                    logits = self.model(x_, attention_mask=extended_attention_mask).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits
                    max_logits = torch.argmax(logits, dim=-1)
                    # print("max logit 0:", max_logits[0][mask_index[0]])
                    # print("max logit 1:", max_logits[-1][mask_index[-1]])
                    # print("logits shape:", logits.shape)
                    # print(logits[-1])

                if temperature > 0.:
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                else:
                    x0 = torch.argmax(logits, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
                elif remasking == 'random':
                    x0_p = torch.rand_like(logits[:, :, 0])
                else:
                    raise NotImplementedError(remasking)

                confidence_mask = torch.ones_like(x, dtype=torch.bool)
                if not is_left:
                    for j in range(batch_size):
                        confidence_mask[j, end_pos[j]:] = False
                else:
                    confidence_mask[:, end_pos:] = False
                
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index & confidence_mask, x0_p, -torch.inf)

            
                num_masked_in_block = block_mask_index.sum(dim=-1)
                num_to_reveal = torch.ceil(num_masked_in_block * (1.0 / steps_per_block)).long()

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    k = min(num_to_reveal[j].item(), (confidence[j] > -torch.inf).sum().item())
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True
                
                x[transfer_index] = x0[transfer_index]
                true_indices = torch.where(transfer_index[-1])[0].tolist()
                
        if not is_left:
            for i in range(batch_size):
                x_generated = x[i, prompt_len[i]:prompt_len[i] + gen_length].clone()
                x[i,-gen_length:] = x_generated.clone()
                x[i, : -gen_length] = prompt[i].clone()
                # x[i, : prompt_len[i]] = prompt[i, :prompt_len[i]].clone()
        # response = self.tokenizer.batch_decode(x[:, -gen_length:], skip_special_tokens=True)
        responses = []
        for i in range(batch_size):
            responses.append(self.tokenizer.decode(x[i, -gen_length:], skip_special_tokens=True))
        print('--------------------')
        for i in range(batch_size):
            print(f'Response {i}:', responses[i])
            print('====================')
        print('--------------------')
        return responses
    @ torch.no_grad()
    def _generate(self, prompt):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        '''
        print('--------------------')
        steps = self.gen_steps
        gen_length = self.gen_length
        block_length = self.gen_blocksize
        cfg_scale = self.cfg
        temperature = self.temperature
        remasking = self.remasking
        mask_id = self.mask_id
        print('parameters:', steps, gen_length, block_length, temperature, cfg_scale, remasking, mask_id)
        messages = _convert_chat_messages(prompt)
        prompt = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
        print('final prompt:', prompt)
        prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids']
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # print(num_transfer_tokens.sum())
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
        response = self.tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)
        print('--------------------')
        print('response:', response)
        print('--------------------')
        return response

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        batch_size = len(inputs)
        if batch_size > 1:
            return self.batch_generate(inputs)
        return self._generate(inputs)
    
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
    def _get_ppl_masked(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
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
        
        # print(repr(inputs[0]))
        print(inputs)
        prefix_index = len(inputs[0]) - len(mask_length[0])
        mask_length = None
        tokenized = self.tokenizer(
            inputs, 
            return_tensors='pt'
        ).to(self.model.device)
        prefix = self.tokenizer(inputs[0][:prefix_index], return_tensors='pt').to(self.model.device)
        batch = tokenized['input_ids']

        if mask_length is not None:
            max_mask_len = max(mask_length)
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
        else:

            max_mask_len = len(prefix['input_ids'][0])
            mask_length = [max_mask_len]
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
            # prompt_index = torch.zeros(batch.shape[1], dtype=torch.bool, device=batch.device)
        
        single_seq = batch.repeat(self.batch_size, 1)
        lens = single_seq.shape[1]
        target_len = (single_seq.shape[1] - mask_length[0])
        
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(single_seq, prompt_index)
            
            mask_indices = perturbed_seq == self.mask_id
            
            logits = self.get_logits(perturbed_seq, prompt_index)
            
            loss = F.cross_entropy(
                logits[mask_indices], 
                single_seq[mask_indices], 
                reduction='none'
            ) / p_mask[mask_indices]
            
            normalized_loss = loss.sum() / (self.batch_size * target_len)

            loss_acc.append(normalized_loss.item())
        
        sample_ppl = sum(loss_acc) / len(loss_acc)
        sample_ppl = [sample_ppl]
        return sample_ppl

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
    def get_answer_prefix_index(self, text: str) -> int:
        """获取答案部分的起始索引"""
        
        # 尝试不同的答案模式
        patterns = [
            ('Answer:', 'Answer:'),
            ('Answer: \n', 'Answer: '),  # 移除换行符
            ('答案:', '答案:'),
            ('答案是:', '答案是:'),
            ('A:', 'A:'),
        ]
        
        for pattern, replacement in patterns:
            index = text.rfind(pattern)
            if index != -1:
                if pattern == 'Answer: \n':
                    # 特殊处理：替换换行符
                    text = text.replace('Answer: \n', 'Answer: ')
                    return text.rfind('Answer:') + len('Answer:')
                else:
                    return index + len(pattern)
        
        # 如果都没找到，使用最后一个换行符（hellaswag的情况）
        print('this is target for hellaswag')
        last_newline = text.rfind('\n')
        return last_newline +1 if last_newline != -1 else 0

    def get_loglikelihood(self, inputs: List[str], conts:  List[str]) -> List[float]:
        print('inputs:', inputs, 'conts:', conts)
        mask_length = [self.get_token_len(c, add_special_tokens=False) for c in conts]
        return - self.get_ppl(inputs, mask_length)
    def _get_ppl(self,
            inputs: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores using diffusion-based Monte Carlo sampling."""
        
        # 1. Tokenize输入
        # print(repr(inputs[0]))
        print(inputs)
        print(mask_length)
        prefix_index = 0
        if mask_length is None:
            inputs[0] = inputs[0].replace('Answer: \n', 'Answer: ')
            prefix_index = self.get_answer_prefix_index(inputs[0])
        ## hellaswag
            inputs[0] = inputs[0][:prefix_index] + ' ' + inputs[0][prefix_index+1:]
        # prefix_index = inputs[0].rfind('Answer:') + len('Answer:') ##arcc 'Answer: \n' 35.x, 'Answer: ' 41.x, 'Answer:' 40.x
        # if prefix_index == -1 + len('Answer:'):
        #     prefix_index = inputs[0].rfind('答案:') + len('答案: \n') ## ceval
        # elif prefix_index == -1 + len('答案: \n'):
        #     prefix_index = inputs[0].rfind('答案是:') + len('答案是:') ## cmmlu. 不知道为什么这里不能 \n
        # elif prefix_index == -1 + len('答案是:'):
        #     prefix_index = inputs[0].rfind('A:') + len('A:')
        # elif prefix_index == -1 + len('A:'):
        #     print('this is target for hellaswag')
        #     prefix_index = inputs[0].rfind('\n') + len('\n')
        # print(inputs[0][prefix_index:])
        # print(repr(inputs[0]))
        tokenized = self.tokenizer(
            inputs, 
            return_tensors='pt'
        ).to(self.model.device)
        if mask_length is None:
            prefix = self.tokenizer(inputs[0][:prefix_index], return_tensors='pt').to(self.model.device)
        batch = tokenized['input_ids']
        # print(len(batch[0]),len(prefix['input_ids'][0]))
        # 2. 计算prompt_index
        if mask_length is not None:
            max_mask_len = batch.shape[1] - max(mask_length)
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
        else:
            # mask_length = [batch.shape[1] - 1]
            # max_mask_len = batch.shape[1] - 1

            max_mask_len = len(prefix['input_ids'][0])
            mask_length = [max_mask_len]
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
            # prompt_index = torch.zeros(batch.shape[1], dtype=torch.bool, device=batch.device)
        
        # 3. 为每个样本单独计算困惑度
        single_seq = batch.repeat(self.batch_size, 1)
        # 计算target部分长度（用于归一化）
        lens = single_seq.shape[1]
        target_len = (single_seq.shape[1] - mask_length[0])
        # target_len = (batch.shape[1] - max_mask_len)
        # print('----------------')
        # print(target_len)
        
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            # 使用扩散前向过程创建掩码版本
            perturbed_seq, p_mask = self._forward_process(single_seq, prompt_index)
            
            # 找到被mask的位置
            mask_indices = perturbed_seq == self.mask_id
            
            # 获取模型预测
            logits = self.get_logits(perturbed_seq, prompt_index)
            
            # 计算掩码位置的交叉熵损失，应用重要性采样权重
            loss = F.cross_entropy(
                logits[mask_indices], 
                single_seq[mask_indices], 
                reduction='none'
            ) / p_mask[mask_indices]
            
            normalized_loss = loss.sum() / (self.batch_size * target_len)
            # all lens
            # normalized_loss = loss.sum() / (lens * self.batch_size)
            # normalized_loss = normalized_loss / target_len
            loss_acc.append(normalized_loss.item())
        
        # 对所有蒙特卡洛样本取平均
        sample_ppl = sum(loss_acc) / len(loss_acc)
        sample_ppl = [sample_ppl]
        return sample_ppl
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



def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    if temperature < 0.01:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

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