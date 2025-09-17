import torch 
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim

class BPRLoss(nn.Module):
    def forward(self, user, neg, pos, weights, criterion):
          
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
    def __init__(self, user_dim, item_dim, embedd_dim, hidden_dim, ocupation_dim, zip_dim):
        super(TwoTowerModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embedd_dim = embedd_dim
        self.ocupation_dim = ocupation_dim
        self.zip_dim = zip_dim

        self.ocupation_embedding = nn.Embedding(self.ocupation_dim, 24)
        self.zip_embedding = nn.Embedding(self.zip_dim, 24)
        self.item_embedding = nn.Embedding(self.item_dim, self.embedd_dim)  # 3000 x 512     x    
        self.user_embedding = nn.Embedding(self.user_dim, self.embedd_dim)
        
  
        self.usertower = torch.nn.Sequential(
            torch.nn.Linear(self.embedd_dim + 24 +  3 + 24, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        #(1x131 and 128x64)

        self.itemtower = torch.nn.Sequential(
            torch.nn.Linear(self.embedd_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )



    
    def forward(self,batch):

    
        s_negative_embedding = self.item_embedding(batch["negs"])
        s_pos_embedding = self.item_embedding(batch["pos"])
        user_embedding = self.user_embedding(batch["userid"])
        user_dense = batch["dense"]
        try:
          user_zip_embedding = self.zip_embedding(batch["zip"])
        except Exception as e:
            
          print(f"{batch['zip']} did not exist in zip  embedding")   
        try:
          user_occupation_embedding = self.ocupation_embedding(batch["occupation"])
        except Exception as e: 
          print(batch["occupation"])  
          print(f"{batch['occupation']} did not exist in occupation embedding")    
          


        



        
      
        user_embedding = torch.cat([user_embedding,
                                    user_dense, 
                                    user_zip_embedding, 
                                    user_occupation_embedding], dim = 1
                                    ) # 1 x 307
        

        #print(user_embedding.shape)

        s_negative_embedding = self.itemtower(s_negative_embedding)

        #print(s_negative_embedding.shape)
        s_positive_embedding = self.itemtower(s_pos_embedding)

        #print(s_positive_embedding.shape)
        s_user_embedding = self.usertower(user_embedding)

        #print(s_user_embedding.shape)
      
     
        return s_negative_embedding, s_positive_embedding, s_user_embedding
