from mmengine.config import read_base
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
        num_worker=8,    
        num_split=None,   
        min_task_size=16, 
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)
