import json

from opencompass.datasets import GPQADataset_, GPQAEvaluator, GPQA_Simple_Eval_postprocess_
# from opencompass.datasets.gpqa import GPQA_Simple_Eval_postprocess
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import (FixKRetriever, RandomRetriever,
                                               ZeroRetriever)
import re

def GPQASimpleEval_postprocess_(text: str) -> str:
    ANSWER_PATTERN = r'(?i)ANSWER\s*:\s*([A-D])'
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1)
    return None

gpqa_reader_cfg = dict(
    input_columns=['question', 'explanation', 'A', 'B', 'C', 'D'],
    output_column='answer')

_hint = """Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.

"""
gpqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=
                f'{_hint}\nQuestion:\n{{question}}\nChoices:\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n'
            ),
            dict(role='BOT', prompt=f'The correct answer is ({{answer}})\n')
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f"Question: {{question}}\nChoices:\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nWhen you're ready to answer, please use the format 'The correct answer is (X)' where X is the correct letter choice from 'ABCD'."
                ),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)

gpqa_eval_cfg = dict(
    evaluator=dict(type=GPQAEvaluator),
    pred_postprocessor=dict(type=GPQA_Simple_Eval_postprocess_)) 

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv',
    # 'experts': 'gpqa_experts.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(abbr='GPQA_' + split,
             type=GPQADataset_,
             path='./data/gpqa/',
             name=gpqa_subsets[split],
             reader_cfg=gpqa_reader_cfg,
             infer_cfg=gpqa_infer_cfg,
             eval_cfg=gpqa_eval_cfg))