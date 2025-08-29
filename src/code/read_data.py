import pandas as pd 
import os 
from urllib.request import urlretrieve
import zipfile
from pathlib import Path


def fetch_file(zip, name, path, delimiter, colnames):
     with zipfile.ZipFile(zip, "r") as z:
          for file in z.namelist():
               if name in file:
                    with z.open(file) as FiLe:
                         df = pd.read_csv(FiLe, delimiter = delimiter, names = colnames, index_col = False,  encoding="latin-1")
     df.to_csv(path)


if __name__ == "__main__":


    url =  "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

    file = "ml-1m.zip"

    save_dir =  "C:/Users/oscwa/recommendation_system/data"

    zip, _ = urlretrieve(url, os.path.join(save_dir, file))


    fetch_file(zip = zip,
                    name = "ratings", 
                    path = "C:/Users/oscwa/recommendation_system/data/ratings.csv", 
                    delimiter = '::', 
                    colnames = ["userid", "movieid", "rating"] 
                )

    fetch_file(zip = zip,
                    name = "movies", 
                    path = "C:/Users/oscwa/recommendation_system/data/movies.csv", 
                    delimiter = '::', 
                    colnames = ["MovieID", "Title", "Genres"] 
                )

    fetch_file(zip = zip,
                    name = "users", 
                    path = "C:/Users/oscwa/recommendation_system/data/users.csv", 
                    delimiter = '::', 
                    colnames = ["UserID", "Gender" , "Age" , "Occupation" , "Zip-code"] 
                )


    
