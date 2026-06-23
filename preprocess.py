import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    df = pd.read_csv("survey.csv")

    # Drop columns with too many nulls or irrelevant ones
    df.drop(columns=['Timestamp', 'comments', 'state', 'Country'], inplace=True, errors='ignore')

    # Fix Age outliers
    df = df[(df['Age'] >= 18) & (df['Age'] <= 65)]

    # Standardize Gender
    df['Gender'] = df['Gender'].str.lower().str.strip()
    male_terms   = ['male', 'm', 'man', 'cis male', 'male (cis)', 'make']
    female_terms = ['female', 'f', 'woman', 'cis female', 'female (cis)']
    df['Gender'] = df['Gender'].apply(
        lambda x: 'Male' if x in male_terms
        else ('Female' if x in female_terms else 'Other')
    )

    # Target column: 'treatment' (Yes/No → 1/0)
    df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})

    # Fill missing values
    df.fillna(df.mode().iloc[0], inplace=True)

    # Encode all remaining categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('treatment', axis=1)
    y = df['treatment']

    return train_test_split(X, y, test_size=0.2, random_state=42)