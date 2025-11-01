import torch.optim as optim
import torch.nn as nn
from collections import defaultdict
import torch 
from model import BPRLoss



class trainer:
       def __init__(self, model, dataloader_train, dataloader_val, dataloader_rank, num_epochs, lr, loss_func, k, save_model):

        self.num_epochs = num_epochs 
        self.lr = lr 
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.dataloader_rank = dataloader_rank 
        self.metrics = defaultdict(list)
        self.loss_func = loss_func
        self.k = k 
        self.score = torch.nn.CosineSimilarity(dim = 1)
        self.save_model = save_model

       def train(self):
           optimizer = optim.Adam(self.model.parameters(), lr= self.lr, weight_decay=5e-3)
         
   

           for epoch in range(self.num_epochs):
               self.curr_epoch = epoch 
               precision, recall = self.rank()

               self.metrics["precision"].append(precision)
               self.metrics["recall"].append(recall)

               print(f"precision is {precision} and recall is {recall} at epoch {epoch}")

               running, nsteps = 0.0, 0
               for batch in self.dataloader_train:

                   optimizer.zero_grad()
                   neg, pos, user = self.model(batch)
                        
                   weight_vector = batch["weights"]


                
                   loss = self.loss_func(user, neg, pos, weight_vector, self.score)
                   loss.backward()
                   optimizer.step()

                   running += loss.item()
                   nsteps += 1
               total_loss = running / max(nsteps,1)
               self.metrics["train_loss"].append(total_loss)
               print(f"epoch [{self.curr_epoch+1}/{self.num_epochs}], loss: {total_loss:.6f}")
               self.val()

           if self.save_model == True:
                torch.save(self.model.state_dict(), "my_model")
             
           return self.metrics
        

       def val(self):
            running, nsteps = 0.0, 0
            with torch.no_grad():
              for batch in self.dataloader_val:
                    neg,pos,user = self.model(batch)
                    weight_vector = batch["weights"]
                    loss = self.loss_func(user, neg, pos, weight_vector, self.score)
                    running += loss.item()
                    nsteps += 1

              total_loss = running / max(nsteps,1)
              self.metrics["val_loss"].append(total_loss)
              print(f"epoch [{self.curr_epoch+1}/{self.num_epochs}], #val_loss#: {total_loss:.6f}")

       def rank(self):

        mean_precision_by_user = []
        mean_recall_by_user = []
        for batch in self.dataloader_rank:
            with torch.no_grad():

                

                neg, pos, user = self.model(batch)

                

                candidates = torch.cat([pos.squeeze(0), neg.squeeze(0)], dim=0)
                

                #user = F.normalize(user, dim=-1) 
                #candidates = F.normalize(candidates, dim =-1)
                scores = torch.matmul(user, candidates.T)/0.2

                

                labels = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
            
                topk_scores, topk_idx = torch.topk(scores, self.k, dim=1)   # (u_dim, k)
                topk_labels = labels[topk_idx] 
                recall_at_k = topk_labels.sum(dim = 1)/ labels.sum()
                precision_at_k = topk_labels.sum(dim=1) / self.k             # (u_dim,)
                mean_precision_at_k = precision_at_k.mean().item()
                mean_precision_by_user.append(mean_precision_at_k)
                mean_recall_by_user.append(recall_at_k)
        precision_at_k = sum(mean_precision_by_user) / len(mean_precision_by_user)
        recall_at_k = sum(recall_at_k) / len(recall_at_k)
        return precision_at_k, recall_at_k  
                