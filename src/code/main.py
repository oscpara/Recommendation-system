from read_data import fetch_file
from model import BPRLoss
from urllib.request import urlretrieve
import os 
import pandas as pd 
from preprocessing import process
from dataclass import userdataset
from model import TwoTowerModel
import torch 
from torch.utils.data import Dataset, DataLoader
from model import BPRLoss
from train import trainer


def main():
   
    ratings = pd.read_csv("C:/Users/oscwa/recommendation_system/data/ratings.csv", index_col = 0)



    process_init = process(ratings = ratings)

    train, val, ratings = process_init.processing()


    dataset_train = userdataset(train, strong_neg_weight=1, weak_neg_weight= 1, k = 20, q = 2, mode = "train" )
    dataset_val = userdataset(train, strong_neg_weight=1, weak_neg_weight= 1, k = 20, q = 2, mode = "train" ) 
    dataset_rank = userdataset(val, strong_neg_weight=1, weak_neg_weight= 1, k = 2, q = 2, mode = "val" )


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


    dataloader_train = DataLoader(dataset_train, batch_size= 32, shuffle= True, drop_last= True)
    dataloader_val = DataLoader(dataset_val, batch_size= 32, drop_last= True, shuffle= True) 
    dataloader_rank = DataLoader(dataset_rank, batch_size= 1, drop_last= True)


    trainer_obj = trainer(model = model, 
            dataloader_train= dataloader_train, 
            dataloader_val= dataloader_val, 
            dataloader_rank= dataloader_rank, 
            num_epochs= 10, 
            lr  = 0.001, 
            loss_func= BPRLoss(),
            k = 10,
            save_model=True)

    trainer_obj.train()



    if __name__ == "__main__":
      main()