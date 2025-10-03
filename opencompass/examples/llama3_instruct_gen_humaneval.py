from mmengine.config import read_base
# from opencompass.configs.datasets.humaneval.internal_humaneval_gen_ce6b06 import \
#     internal_humaneval_gen
with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import \
        humaneval_datasets
    # from opencompass.configs.datasets.humaneval.internal_humaneval_gen_d2537e import \
    #     humaneval_datasets
    from opencompass.configs.models.hf_llama.hf_llama3_8b_instruct import \
        models as hf_llama3_8b_instruct_models
    # from opencompass.configs.models.dllm.llada_chat_8b import \
    #     models as llada_instruct_8b_models
    # from opencompass.configs.summarizers.groups.humaneval import \
    #     humaneval_summary_groups
datasets = humaneval_datasets
models = hf_llama3_8b_instruct_models

# summarizer = dict(
#     summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
# )

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,    # 划分完成后的任务数 / 预期能有的 worker 数
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