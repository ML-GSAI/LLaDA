from mmengine.config import read_base
# /home/xushaoxuan/d-llm/opencompass/opencompass/configs/datasets/mmlu/mmlu_gen_a484b3.py
# answer = 256, block = 256, fewshot = 0
with read_base():
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import \
        mmlu_pro_datasets
    from opencompass.configs.models.hf_llama.hf_llama3_8b_instruct import \
        models as hf_llama3_8b_instruct_models
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups
datasets = mmlu_pro_datasets
models = hf_llama3_8b_instruct_models
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=7,    # 划分完成后的任务数 / 预期能有的 worker 数
        num_split=None,   # 每个数据集将被划分成多少份。若为 None，则使用 num_worker。
        min_task_size=16, # 每个划分的最小数据条目数
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)
# python run.py examples/llada_gen_bbh.py -w outputs/demo