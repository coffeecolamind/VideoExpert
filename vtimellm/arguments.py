from dataclasses import dataclass, field
from typing import Optional, List
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    stage2_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    loc_interaction_type: Optional[str] = field(default='simp_add')
    patch_filter_type: Optional[str] = field(default='only_cls')

    ce_loss_weight: float = field(default=1.0,)
    det_loss_weight: float = field(default=1.0,)
    span_loss_type: Optional[str] = field(default="l1")
    det_loss_b: float = field(default=5.0,)  ## loss_span - F.smooth_l1_loss
    det_loss_g: float = field(default=2.0,)  ## loss_giou
    det_loss_f: float = field(default=2.0,)  ## loss_ce

    clip_path: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_stage: int = field(default=2)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    tune_mm_mlp_stage2: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


@dataclass
class DataArguments:
    task_sample_rate: list[float] = None
    pretrain_sample_rate: list[float] = None
    temp_loc_sample_rate: list[float] = None
    dvc_sample_rate: list[float] = None
    videoqa_sample_rate: list[float] = None

    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    feat_folder: Optional[str] = field(default=None)

    tasks: str = 'pretrain'  # 'event_loc||imgqa||vidqa'
    pretrain_data: str = 'Stage2||Stage3'  # 'Stage2||Stage3'
    dvc_data: str = 'Youcook2'  # 'ActivityNet||Youcook2'
    temp_loc_data: str = 'QVHL'  # 'QVHL||Charades'
    videoqa_data: str = 'NextQA'  # 'NextQA||TVQA'
    temporal_reasoning_data: str = 'activitynet'
    samples_per_epoch: int = 50000 # 500 * 8 * 2 * 10
    use_RAG: bool = False
    RAG_path: str = field(default=None, metadata={"help": "Path to the training data."})

