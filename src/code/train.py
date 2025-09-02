import torch.optim as optim
import torch.nn as nn
import torch 

class BPRLoss(nn.Module):
    def forward(self, user, neg, pos, weights, criterion):
            cosine_vector_neg = criterion(user.unsqueeze(1), neg)
            cosine_vector_pos = criterion(user, pos).unsqueeze(1)

            return ((-(torch.log(torch.sigmoid(cosine_vector_pos - cosine_vector_neg))))*weights).mean()
        
 


class train:
       def __init__(self, model, dataloader, num_epochs, lr):

        self.num_epochs = num_epochs 
        self.lr = lr 
        self.model = model
        self.dataloader = dataloader 
        

       def train(self):

        optimizer = optim.Adam(self.model.parameters(), lr= self.lr)
        score = torch.nn.CosineSimilarity(dim = -1)
        loss_fn = BPRLoss()
        train_loss = []
        for epoch in range(self.num_epochs):
                running, nsteps = 0.0, 0
                for u, pos, negs, weights in self.dataloader:
                        optimizer.zero_grad()
                        neg, pos, user = self.model(u, pos, negs)

                        weight_vector = weights


                
                        loss = loss_fn(user, neg, pos, weight_vector, score)
                        loss.backward()
                        optimizer.step()
        
                        running += loss.item()
                        nsteps += 1
                total_loss = running / max(nsteps,1)
                train_loss.append(total_loss)
                print(f"epoch [{epoch+1}/{self.num_epochs}], loss: {total_loss:.6f}")