from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.math.math_gen import \
        math_datasets
    from opencompass.configs.models.dllm.llada_base_8b import \
        models as llada_base_8b_models
    from opencompass.configs.summarizers.groups.mathbench import \
        mathbench_summary_groups
datasets = math_datasets
models = llada_base_8b_models
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
eval_cfg = {'gen_blocksize': 256, 'gen_length': 256, 'gen_steps': 256, 'stop_words': ['Problem:']}
for model in models:
    model.update(eval_cfg)
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
