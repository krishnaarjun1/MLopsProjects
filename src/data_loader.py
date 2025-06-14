import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv("C:/Users/HP/Desktop/MLopsProject/Titanic_Project/Data/titanic_data.csv")
