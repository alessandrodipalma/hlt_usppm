from pytorch_lightning import LightningDataModule
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from USPPM_dataset import USPPM_dataset

class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass
    
@dataclass
class USPPPM_kf_datamodule(BaseKFoldDataModule):
    def __init__(self, config_dict, dataframe):
        
        self.config_dict = config_dict
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        
        train_dataset: Optional[Dataset] = None
        test_dataset: Optional[Dataset] = None
        train_fold: Optional[Dataset] = None
        val_fold: Optional[Dataset] = None
        
        self.dataframe = dataframe
            
    def setup(self, stage: Optional[str] = None) -> None:
        # train_df, test_df = train_test_split(self.dataframe, test_size = 0.1, random_state = CFG.seed, stratify = self.dataframe.score_map)
        train_df, test_df = train_test_split(self.dataframe, test_size = 0.1, random_state = self.config_dict["seed"])
        self.train_dataset = USPPM_dataset(self.config_dict, train_df)
        self.test_dataset = USPPM_dataset(self.config_dict, test_df)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        Fold = StratifiedKFold(n_splits=self.num_folds, shuffle=True)
        self.splits = [split for split in Fold.split(self.train_dataset, self.train_dataset.score_map)]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)
        print("TRAIN FOLD", fold_index + 1, len(self.train_fold))
        print("VALID FOLD", fold_index + 1, len(self.val_fold))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, num_workers = self.config_dict['num_workers'], batch_size = self.config_dict['batch_size'])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, num_workers = self.config_dict['num_workers'], batch_size = self.config_dict['batch_size'])
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, num_workers = self.config_dict['num_workers'], batch_size = self.config_dict['batch_size'])
    
    def __post_init__(cls):
        super().__init__()