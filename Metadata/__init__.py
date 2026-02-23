from pydantic import BaseModel
from enum import StrEnum
from typing import List,Tuple,Optional,Dict,Any
from omegaconf import ListConfig
from Utils.FeatureExtractor.Base import BaseExtractor

class FileLoaderMetadata(BaseModel):
    name: str
    path: str
    sample_needed:Optional[bool] = False
    sampling_rate: Optional[int] = None
    file_names : Optional[List[str]] = None

class CSVLoaderMetadata(BaseModel):
    rri_csv_path: str
    features_csv_path: str

class RRSequenceMetadata(BaseModel):
    window_size: int
    stride: int
    horizons: list[int]
    seq_len: int

class FeatureType(StrEnum):
    LFHF = 'lfhf'
    RMSSD = 'rmssd'
    EctopicPercentage = 'ectopic_percentage'
    Alpha1 = 'alpha1'
    SampleEntropy = 'sample_entropy'
    ApproximateEntropy = 'approximate_entropy'

class LossType(StrEnum):
    MSE = 'mse'
    MAE = 'mae'

class OptimizerType(StrEnum):
    ADAMW = 'adamW'
    ADAM = 'adam'

class FeatureMetadata(BaseModel):    
    list: List[Dict[FeatureType, Optional[dict]]]

    def convert(self) -> List[Tuple[FeatureType, Optional[Dict]]]:
        converted_list = []
        for item in self.list:
            for feature_type_str, params in item.items():
                feature_type = FeatureType(feature_type_str)
                converted_list.append((feature_type, params or {}))
        return converted_list

    

class DatasetMetadata(BaseModel):
    name: str
    sampling_frequency: int
    file_loader: FileLoaderMetadata
    # rr_sequence: RRSequenceMetadata
    # feature_types: Any  # Will be instantiated by RRSequenceDataset using hydra.utils.instantiate


class SplitMetadata(BaseModel):
    train:float
    val:float
    test:float

    def __post_init__(self):
        total = self.train + self.val + self.test
        if total != 1.0:
            raise ValueError("Train, validation, and test splits must sum to 1.0")

class PreprocessingMetadata(BaseModel):
    rr_sequence: RRSequenceMetadata        

class EncoderConfig(BaseModel):
    latent_dim: int

class ARBlockConfig(BaseModel):
    latent_dim: int
    context_dim: int

class HRVPredictorConfig(BaseModel):
    context_dim: int
    num_targets: int
    num_heads: int
class CPCPreModelConfig(BaseModel):
    encoder: EncoderConfig
    ar: ARBlockConfig
    predictor: HRVPredictorConfig
# epochs: 100
# loader:
#   shuffle: true
#   batch_size: 32
#   pin_memory: true
# optimizer:
#   name: adamW
#   lr: 0.001
# loss:
#   name: mse
#   weights: [1.0, 1.0, 1.0]

class LoaderConfig(BaseModel):
    shuffle: bool
    batch_size: int
    pin_memory: bool
    num_workers: int = 0

class OptimizerConfig(BaseModel):
    name: OptimizerType
    lr: float

class LossConfig(BaseModel):
    name: LossType
    weights: Optional[List[float]] = None
class TrainerConfig(BaseModel):
    epochs: int
    loader: LoaderConfig
    optimizer: OptimizerConfig
    loss: LossConfig

class ValidatorConfig(BaseModel):
    loader: LoaderConfig
    loss: LossConfig

class TesterConfig(BaseModel):
    loader: LoaderConfig
    loss: LossConfig

class Config(BaseModel):
    seed : int
    split: SplitMetadata
    device: str
    dataset: FileLoaderMetadata
    features: FeatureMetadata
    preprocessing: PreprocessingMetadata
    model : CPCPreModelConfig
    trainer: TrainerConfig
    validator: ValidatorConfig
    tester: TesterConfig