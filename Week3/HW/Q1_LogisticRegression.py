import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

dataset = pd.read_csv(r"Week3\HW\Datasets\Q1_admission_result.csv")
#print(dataset.head())

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]



