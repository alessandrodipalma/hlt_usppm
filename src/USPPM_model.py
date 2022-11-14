from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
import numpy as np
import scipy as sp


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score

class USPPPM_model(LightningModule):
    def __init__(self, config_dict, config_path=None, pretrained=True):
        super().__init__()
        
        if config_path is None:
            self.config = AutoConfig.from_pretrained(config_dict['model'], output_hidden_states = True)
        else:
            self.config = torch.load(config_path)
        
        if pretrained:
            self.model = AutoModel.from_pretrained(config_dict['model'], config = self.config)
        else:
            self.model = AutoModel.from_config(self.config)
            
        self.config_dict = config_dict
        self.n_warmup_steps = config_dict['warmup_steps']
        self.n_training_steps = config_dict['training_steps']
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
            
        self.fc_dropout = nn.Dropout(config_dict['fc_dropout'])
        self.fc = nn.Linear(self.config.hidden_size, config_dict['target_size'])
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        
        self.batch_labels = []
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs, labels=None):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        loss, outputs = self(inputs, labels.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        # session.report({"train_loss": loss})  # Send the score to Tune.
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        loss, outputs = self(inputs, labels.unsqueeze(1))
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # session.report({"val_loss": loss})  # Send the score to Tune.
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        loss, outputs = self(inputs, labels.unsqueeze(1))
        self.log("test_loss", loss, prog_bar=True, logger=True)
        # session.report({"test_loss": loss})  # Send the score to Tune.
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def validation_epoch_end(self, batch_results):
        outputs, labels, losses = [], [], []
        for batch in batch_results:
            outputs.append(batch['predictions'])
            labels.append(batch['labels'])
            losses.append(batch['loss'])

        labels = torch.cat(labels).cpu().numpy()
        predictions = np.concatenate(torch.cat(outputs).sigmoid().to('cpu').numpy())
        score = get_score(labels, predictions)
        self.log("val_score", score, prog_bar=True, logger=True)
        # tune.report({"val_score": score})  # Send the score to Tune.

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config_dict['encoder_lr'])
        # optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )

'''        def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if "model" not in n],'lr': decoder_lr, 'weight_decay': 0.0}
            ]
            return optimizer_parameters

        optimizer_parameters = get_optimizer_params(self,
                                            encoder_lr=self.config_dict["encoder_lr"], 
                                            decoder_lr=self.config_dict["decoder_lr"],
                                            weight_decay=self.config_dict["weight_decay"])

        optimizer = AdamW(optimizer_parameters, 
                          lr=self.config_dict["encoder_lr"], 
                          eps=self.config_dict["eps"],
                          betas=self.config_dict["betas"])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )'''

