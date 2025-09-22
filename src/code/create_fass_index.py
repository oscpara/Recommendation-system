import faiss 
import numpy as np 


user_embeddings_file = np.load("user_embeddings.npz")
item_embeddings_file = np.load("item_embeddings.npz")


user_embeddings = user_embeddings_file["emb"]
item_embeddings = item_embeddings_file["emb"]

user_ids = user_embeddings_file["user_ids"]
item_ids = item_embeddings_file["item_ids"]



base = faiss.IndexFlatIP(item_embeddings.shape[1])               
index = faiss.IndexIDMap(base) 


index.add_with_ids(item_embeddings, item_ids) 


faiss.write_index(index, "faiss_indices_items.index")



