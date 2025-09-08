import torch 
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

class BPRLoss(nn.Module):
    def forward(self, user, neg, pos, weights):
          
            tau = 0.2

            s_pos = ((user * pos)/tau).sum(dim=-1).unsqueeze(1)
                     # [1]
            s_neg = ((user.unsqueeze(1) * neg)/tau).sum(dim=-1)
                      # [4]
            
            x = s_pos  - s_neg  

            loss_vec = F.softplus(-x)                # [4]
            loss = (loss_vec * weights).sum() / (weights.sum() + 1e-8)


            return loss 
        
 


class TwoTowerModel(nn.Module):
    def __init__(self, user_dim, item_dim, embedd_dim, hidden_dim):
        super(TwoTowerModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embedd_dim = embedd_dim

        self.item_embedding = nn.Embedding(self.item_dim, self.embedd_dim)  # 3000 x 512     x    
        self.user_embedding = nn.Embedding(self.user_dim, self.embedd_dim)


        self.usertower = torch.nn.Sequential(
            torch.nn.Linear(self.embedd_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.itemtower = torch.nn.Sequential(
            torch.nn.Linear(self.embedd_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
    
    def forward(self, users, pos, neg):
        s_negative_embedding = self.item_embedding(neg)
        s_pos_embedding = self.item_embedding(pos)
        user_embedding = self.user_embedding(users)
        return s_negative_embedding, s_pos_embedding, user_embedding
