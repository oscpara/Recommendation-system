from torch.utils.data import Dataset, DataLoader
import torch 
from collections import defaultdict
import numpy as np 


class userdataset(Dataset):
    def __init__(self, data, strong_neg_weight, weak_neg_weight, k):
        self.data = data 
        self.positive_by_user, self.negative_by_user = self.create_lookup()
        self.strong_neg_weight = strong_neg_weight
        self.weak_neg_weight = weak_neg_weight
        self.k = k
        self.users = set(self.data["userid"])
        
    
    def sample_strong_negatives(self, user_id):
         user_negative_pairs = self.negative_by_user[user_id]
         if user_negative_pairs:
           item = np.random.choice(list(self.negative_by_user[user_id]))
           return item, self.strong_neg_weight
         else:
           return self.sample_weak_negatives(user_id)
         
    def sample_weak_negatives(self,user_id):

         existing_items = self.data["movieid"].unique().tolist()
         while True:
           item = np.random.choice(existing_items)
           if (item not in self.positive_by_user[user_id]) and (item not in  self.negative_by_user[user_id]):
             break 
         return item, self.weak_neg_weight
             
         
    
    def sample_positive(self, user_id):
        try:
          item = np.random.choice(list(self.positive_by_user[user_id]))
        except Exception as e:
            return print(e)
        return item 
         
    def create_lookup(self):
        
          users_positive_pairs = defaultdict(set) 
          users_negative_pairs = defaultdict(set)
        
          for row in self.data.itertuples(index= False):
            if row.label == 1:
                users_positive_pairs[row.userid].add(row.movieid)
            else:
                users_negative_pairs[row.userid].add(row.movieid)

            return users_negative_pairs, users_positive_pairs
          
    def get_k_negatives(self, user_id):
        i = 0
        negs = []
        weights = []
        strong_neg, strong_weight = self.sample_strong_negatives(user_id)
       

        while i <= self.k:
            weak_neg, weak_weight = self.sample_weak_negatives(user_id)
            negs.append(weak_neg)
            weights.append(weak_weight)
            i += 1
        negs.append(strong_neg)
        weights.append(strong_weight)
        return negs, weights 
            
    def __len__(self):
        return len(self.users)                       
        
          
    def __getitem__(self, user_id):
        pos = self.sample_positive(user_id)
        negs, weights = self.get_k_negatives(user_id)
        return (torch.tensor(pos), 
                torch.tensor(negs, dtype= torch.long), 
                torch.tensor(weights, dtype = torch.long)
                )
    