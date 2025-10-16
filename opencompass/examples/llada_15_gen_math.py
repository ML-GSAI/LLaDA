from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.math.math_0shot_gen_11c4b5 import \
        math_datasets
    from opencompass.configs.models.dllm.llada_15_instruct_8b import \
        models as llada_instruct_8b_models
    from opencompass.configs.summarizers.groups.mathbench import \
        mathbench_summary_groups
datasets = math_datasets
models = llada_instruct_8b_models
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
eval_cfg = {'gen_blocksize': 128, 'gen_length': 1024, 'gen_steps': 1024, 'batch_size':1, 'batch_size_':1, 'diff_confidence_eos_eot_inf': True, 'diff_logits_eos_inf': False}
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

