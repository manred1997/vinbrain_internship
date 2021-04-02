import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel

class AcrBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()

        self.bert = BertModel(config, add_pooling_layer=False)

        self.dropout_1 = nn.Dropout(p=0.2)
        self.dense_1  = nn.Linear(768*2, 128)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.1)
        self.dense_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
    
    def forward(
        self, 
        input_ids=None, 
        token_type_ids=None, 
        attention_mask=None, 
        start_token_idx=None, 
        end_token_idx=None):

        features_extract = self.bert(input_ids = input_ids, 
                                    attention_mask = attention_mask, 
                                    token_type_ids = token_type_ids)[0]
        features_cls = features_extract[:, 0, :].unsqueeze(1)
        if start_token_idx is not None and end_token_idx is not None:
            list_mean_feature_acr = []
            for idx in range(features_extract.size()[0]):
                feature_acr = features_extract[idx, start_token_idx[idx]:end_token_idx[idx]+1, :].unsqueeze(0)
                mean_feature_acr = torch.mean(feature_acr, 1, True)
                list_mean_feature_acr.append(mean_feature_acr)
        features_arc = torch.cat(list_mean_feature_acr, dim=0)
        features_in = torch.cat([features_cls, features_arc], dim=2)
        
        features = self.dropout_1(features_in)
        features = self.dense_1(features)
        features = self.relu(features)
        features = self.dropout_2(features)
        features = self.dense_2(features).view(-1)
        
        output = self.sigmoid(features)

        return output, features_in