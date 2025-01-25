import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load_house_attributes(inputPath):
    cols = ['bedrooms', 'bathrooms', 'area', 'zipcode', 'price']
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df['zipcode'].value_counts().tolist()

    for (zipcode, count) in zip(zipcodes, counts):
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)
    
    return df


df = load_house_attributes(r"Week4\Datasets\HousesInfo.txt")

train, test = train_test_split()



