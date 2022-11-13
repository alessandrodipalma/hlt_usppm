from transformers import AutoTokenizer, AutoModel, AutoConfig
# PyTorch
import torch

from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningModule
from tqdm.auto import tqdm

def set_tokenizer(config_dict, OUTPUT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(config_dict['model'])
    tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    config_dict['tokenizer'] = tokenizer
    
    
def set_max_len(config_dict, cpc_texts, train_df):
    tokenizer = config_dict['tokenizer']
    lengths_dict = {}

    lengths = []
    tk0 = tqdm(cpc_texts.values(), total=len(cpc_texts))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    lengths_dict['context_text'] = lengths

    for text_col in ['anchor', 'target']:
        lengths = []
        tk0 = tqdm(train_df[text_col].fillna("").values, total=len(train_df))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            lengths.append(length)
        lengths_dict[text_col] = lengths

    config_dict['max_len'] = max(lengths_dict['anchor']) + max(lengths_dict['target'])\
                    + max(lengths_dict['context_text']) + 4 # CLS + SEP + SEP + SEP
    
    
def prepare_input(config_dict, text):
    tokenizer = config_dict['tokenizer']
    inputs = tokenizer(text,
                       add_special_tokens = True,
                       max_length = config_dict['max_len'],
                       padding = "max_length",
                       return_offsets_mapping = False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class USPPM_dataset(Dataset):
    def __init__(self, config_dict, train_df, train=True):
        self.config_dict = config_dict
        self.texts = train_df['text'].values
        self.train = train
        if train:
            self.labels = train_df['score'].values
            self.score_map = train_df['score_map'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.config_dict, self.texts[item])
        if self.train:
            labels = torch.tensor(self.labels[item], dtype=torch.float)
            return dict(
                  inputs = inputs,
                  labels = labels
            )
        else:
            return dict(
                  inputs = inputs
            )