import torch 
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim, embedd_dim):
        super(TwoTowerModel, self).__init__()

        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embedd_dim = embedd_dim

        self.item_embedding = nn.Embedding(self.item_dim, self.embedd_dim)
        self.user_embedding = nn.Embedding(self.user_dim, self.embedd_dim)
    
    def forward(self, users, pos, neg):
        s_negative_embedding = self.item_embedding(neg)
        s_pos_embedding = self.item_embedding(pos)
        user_embedding = self.user_embedding(users)
        return s_negative_embedding, s_pos_embedding, user_embedding
