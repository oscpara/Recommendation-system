import pandas as pd 


# Read data  

class process:
    def __init__(self, ratings):
       
        self.ratings = ratings

    def processing(self):
            

        # Ratings 


        ratings = pd.read_csv("C:/Users/oscwa/recommendation_system/data/ratings.csv", index_col = 0)
        items = pd.read_csv("C:/Users/oscwa/recommendation_system/data/movies.csv", index_col = 0)
        users =  pd.read_csv("C:/Users/oscwa/recommendation_system/data/users.csv", index_col = 0)

        ratings = ratings[ratings["rating"] != 3]
        # get time period of ratings 

        print(users)

        ratings["date"] = pd.to_datetime(ratings["timestamp"], unit = "s")
        ratings["year"] = ratings["date"].dt.year

        ratings = ratings.merge(users, right_on = "UserID", left_on = "userid", how = "inner")


        ratings["Gender"] = [1 if i == "F" else 0 for i in ratings["Gender"]]
        ratings["Age"] = (ratings["Age"] - ratings["Age"].mean()) / ratings["Age"].std()



        
        ratings["rank"] = ratings.sort_values(["userid", "timestamp"]).groupby("userid").cumcount() + 1




        ratings = pd.get_dummies(data = ratings, prefix= "Gender", columns = ["Gender"], dtype= int)



        #ratings.sort_values(["userid", "timestamp"])

        max_ranks = ratings.groupby("userid")["rank"].max().reset_index().rename(columns= {'rank': 'max_rank'})


        ratings = ratings.merge(max_ranks, on = "userid")




        ratings["is_val"] = (ratings["rank"] >= round(ratings["max_rank"]*0.8)).astype(int)

        ratings["Zip-code"] = ratings["Zip-code"].str.replace("-", "")

        ratings["Zip-code"] = ratings["Zip-code"].astype(int)

        ratings["label"] = [1 if i > 2 else 0 for i in ratings["rating"]]




        # keep only users with ≥ 1 positive rating
        #pos_counts = ratings.groupby(["userid", "is_val"])["label"].sum().reset_index()




        #test = pos_counts.groupby("userid")["label"].transform(lambda x: (x > 0).all())


        #pos_counts[test]

        cond = ratings[ratings["label"] == 1].groupby("userid")["is_val"].nunique() == 2

        # We need to keep users that have atleast 1 positive rating in both train and val 

        # Users that satisfy the condition
        valid_users = cond[cond].index

        # Filter dataframe
        ratings = ratings[ratings["userid"].isin(valid_users)]



        ratings["user_idx"] = ratings["userid"].astype("category").cat.codes
        ratings["item_idx"] = ratings["movieid"].astype("category").cat.codes


        train = ratings[ratings["is_val"] == 0]
        val = ratings[ratings["is_val"] == 1]

        return train, val, ratings 