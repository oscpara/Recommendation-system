import torch 
from preprocessing import process
from dataclass import userdataset
import pandas as pd 
from model import TwoTowerModel
from torch.utils.data import Dataset, DataLoader
import numpy as np 


ratings = pd.read_csv("C:/Users/oscwa/recommendation_system/data/ratings.csv", index_col = 0)


process_init = process(ratings = ratings)
  
train, val, ratings = process_init.processing()


  
item_dim = len(set(ratings["item_idx"]))
user_dim = len(set(ratings["user_idx"]))
occupation_dim = len(set(ratings["Occupation"]))
zip_dim = len(set(ratings["Zip-code"]))


model = TwoTowerModel(user_dim= user_dim,
                    item_dim= item_dim, 
                    embedd_dim= 512, 
                    hidden_dim = 128, 
                    zip_dim= zip_dim, 
                    ocupation_dim= occupation_dim
                    )


model.load_state_dict(torch.load("my_model"))


dataset_retrieval = userdataset(ratings, strong_neg_weight=1, weak_neg_weight= 1, k = 2, q = 2, mode = "retrieval" )


dataloader_retrieval = DataLoader(dataset_retrieval, batch_size= user_dim, shuffle= False, drop_last= False)


batch = next(iter(dataloader_retrieval))


user_embeddings = model.user_embedding.weight.data
item_embedding = model.item_embedding.weight.data

zip_embedding = model.zip_embedding(batch["zip"])
dense_embedding = batch["dense"]
occupation_embeddings = model.ocupation_embedding(batch["occupation"])


user_embedding = torch.cat([user_embeddings,
                            dense_embedding, 
                            zip_embedding, 
                            occupation_embeddings],
                            dim = 1
                        )
with torch.no_grad():
 user_embedding = model.usertower(user_embedding)
 item_embedding = model.itemtower(item_embedding)

#torch.save(user_embedding, "user_embeddings.pt")


user_ids = np.array(list(dataset_retrieval.users))
item_ids = np.array(list(dataset_retrieval.items))

np.savez("user_embeddings.npz", user_ids=user_ids, emb=user_embedding.cpu().numpy())
np.savez("item_embeddings.npz", item_ids=item_ids, emb=item_embedding.cpu().numpy())
# Import ratings
# Run the same preprocessing script 
# 

#model.zip_embedding.weight.data