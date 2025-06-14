from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def preprocess_data(df: pd.DataFrame):
    df = df.drop(['Cabin', 'Name', 'Ticket'], axis=1)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    label_cols = ['Sex', 'Embarked']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
