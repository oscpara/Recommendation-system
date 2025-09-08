import pandas as pd 


# Read data  

class process:
    def __init__(self, ratings):
       
        self.ratings = ratings

    def processing(self):
     
        ratings = self.ratings[self.ratings["rating"] != 3]


        # get time period of ratings 

        ratings["date"] = pd.to_datetime(ratings["timestamp"], unit = "s")
        ratings["year"] = ratings["date"].dt.year

        # Create temporal train and val split 

        ratings["rank"] = ratings.sort_values(["userid", "timestamp"]).groupby("userid").cumcount() + 1

        max_ranks = ratings.groupby("userid")["rank"].max().reset_index().rename(columns= {'rank': 'max_rank'})

        ratings = ratings.merge(max_ranks, on = "userid")


        ratings["is_val"] = (ratings["rank"] >= round(ratings["max_rank"]*0.8)).astype(int)


        ratings["label"] = [1 if i > 2 else 0 for i in ratings["rating"]]


        cond = ratings[ratings["label"] == 1].groupby("userid")["is_val"].nunique() == 2
        
        valid_users = cond[cond].index

        ratings = ratings[ratings["userid"].isin(valid_users)]



        ratings["user_idx"] = ratings["userid"].astype("category").cat.codes
        ratings["item_idx"] = ratings["movieid"].astype("category").cat.codes


        train = ratings[ratings["is_val"] == 0]
        val = ratings[ratings["is_val"] == 1]

        return train, val, ratings 