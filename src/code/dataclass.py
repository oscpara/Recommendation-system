from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np 
import random 
import torch 



class userdataset(Dataset):
    def __init__(self, data, strong_neg_weight, weak_neg_weight, k, q, mode):
        self.data = data
        self.users = set(self.data["user_idx"])
        self.items = set(self.data["item_idx"])
        self.positive_by_user, self.negative_by_user = self.create_lookup()
        self.strong_neg_weight = strong_neg_weight
        self.weak_neg_weight = weak_neg_weight
        self.mode = mode
        self.k = k
        self.q = q

    def sample_strong_negatives(self, negative_interactions, non_interactions):
         if negative_interactions:
            arr_len = len(negative_interactions)
            if arr_len >= 3:
                 items = random.sample(sorted(negative_interactions), 3)
                 return items, [self.strong_neg_weight] * len(items)           
            else:
                items_strong_neg =  random.sample(sorted(negative_interactions), arr_len)
                items_weak_neg, weights_weak_neg = self.sample_weak_negatives(non_interactions, 3 - arr_len)
                weights_strong_neg = [self.strong_neg_weight] * len(items_strong_neg)
             

                return items_strong_neg + items_weak_neg, weights_strong_neg + weights_weak_neg
           
         else:
          
           return self.sample_weak_negatives(non_interactions, 3)
         
    def sample_weak_negatives(self,user_non_interactions, samples):
            
            

            if samples >= len(user_non_interactions):
                items = list(user_non_interactions)
            else:            
                items = random.sample(sorted(user_non_interactions), samples)  

            return items, [self.weak_neg_weight] * len(items)
             
    def interactions_by_user(self, user_id):
        positive_interactions = self.positive_by_user[user_id]
        negative_interactions = self.negative_by_user[user_id]
        
        all_interactions = positive_interactions.union(negative_interactions)
        non_interactions =  self.items.difference(all_interactions)
        return non_interactions, negative_interactions, positive_interactions
    
    def sample_positive(self, user_id):
           try:
             test = np.random.choice(list(self.positive_by_user[user_id]))
             return test 
           except Exception as e:
               print(f"{user_id} does not exist")
         
    def create_lookup(self):
        
          users_positive_pairs = defaultdict(set) 
          users_negative_pairs = defaultdict(set)
     
        
          for row in self.data.itertuples(index= False):
            if row.rating > 2:
                users_positive_pairs[row.user_idx].add(row.item_idx)
            else:
                users_negative_pairs[row.user_idx].add(row.item_idx)
            

          return users_positive_pairs, users_negative_pairs
            
    def __len__(self):
        return len(self.users)                       
        
          
    def __getitem__(self, user_id):


        non_interactions, negative_interactions, positive_interactions = self.interactions_by_user(user_id)

        if self.mode == "train":
           pos = self.sample_positive(user_id)
           negs_weak, negs_weak_weights = self.sample_weak_negatives(non_interactions, samples= self.k)
           negs_strong, negs_strong_weights = self.sample_strong_negatives(negative_interactions, non_interactions)

           negs = negs_weak + negs_strong
           weights = negs_weak_weights + negs_strong_weights

           return (torch.tensor(user_id),
                torch.tensor(pos), 
                torch.tensor(negs, dtype= torch.long), 
                torch.tensor(weights, dtype = torch.long)
                )
        if self.mode == "val":

            pos = list(positive_interactions)
            
            negs_weak= list(non_interactions)
            weights_weak = [self.weak_neg_weight] * len(negs_weak)
             
            negs = negs_weak + list(negative_interactions)

            #weights = ([self.strong_neg_weight] * len(list(negative_interactions))) + negs_weak_weights
             

            return (torch.tensor(user_id),
                torch.tensor(pos, dtype = torch.long), 
                torch.tensor(negs_weak, dtype= torch.long), 
                torch.tensor(weights_weak, dtype = torch.long)
                )
    


        
        
     
