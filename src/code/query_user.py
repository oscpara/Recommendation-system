import faiss
import numpy as np 


class retrieval:
    def __init__(self):

        self.index = faiss.read_index("faiss_indices_items.index")
        self.user_embeddings = np.load("user_embeddings.npz")

    
    def get_user_embedding(self, userid):
        return self.user_embeddings["emb"][int(userid)]

    
    def query(self, userid, k):

        embedding = self.get_user_embedding(userid).reshape(1, -1)


        D, I = self.index.search(embedding, k = k)

        return [{"rank": int(rank), 
                 "similarity": float(d), 
                 "item": int(i)}  
                 for row_d, row_i in 
                 zip(D, I)  
                 for rank, (d, i) in 
                 enumerate(zip(row_d, row_i), start=1)]
    





if __name__ == "__main__":

    init = retrieval()

    print(init.query(80, 10))



