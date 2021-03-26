import torch
import torch.nn as nn

class AcrBertModel(nn.Module):
    def __init__(self, Bertbase):

        self.transformers = Bertbase

        self.dropout_1 = nn.Dropout(p=0.2)
        self.dense_1  = nn.Linear(768*2, 128)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.1)
        self.dense_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, token_type_ids, attention_mask, start_token_idx, end_token_idx):

        features_extract = self.transformers(input_ids = input_ids, 
                                            attention_mask = attention_mask, 
                                            token_type_ids = token_type_ids)[0]

        features_cls = features_extract[:, 0, :].unsqueeze(1)
        if start_token_idx is not None and end_token_idx is not None:
            list_mean_feature_acr = []
            for idx in range(start_token_idx.shape[0]):
                feature_acr = features_extract[idx, start_token_idx[idx]:end_token_idx[idx], :].unsqueeze(0)
                mean_feature_acr = torch.mean(feature_acr, 1, True)
                list_mean_feature_acr.append(mean_feature_acr)
        features_crc = torch.cat(list_mean_feature_acr, dim=0)

        features = torch.cat([features_cls, features_crc], dim=2)

        features = self.dropout_1(features)
        features = self.dense_1(features)
        features = self.relu(features)
        features = self.dropout_2(features)
        features = self.dense_2(features)
        output = self.sigmoid(features)

        return output