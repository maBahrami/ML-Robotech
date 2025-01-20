import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r"Week2\Week2-Datasets\student_score.csv.txt")

#print(dataset.info())
#print(dataset.head())


x = dataset.iloc[:, :-1]   # [:, 0] gives Series while [:, -1] gives DataFrame. input data must be two dimensional
y = dataset.iloc[:, 1]
#print(type(x))
#print(x.shape())

plt.scatter(x, y)
plt.title("hours/scores")
plt.xlabel("hours")
plt.ylabel("score")
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# since there is only one feature, normalizing will not have any effects; although, the equation of the line will be changed
'''
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
'''


regressor = LinearRegression()
regressor.fit(x_train, y_train)

m = round(regressor.coef_[0], 2)
b = round(regressor.intercept_, 2)
print(f"y = {m}x + {b}")


y_pred = regressor.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df)


print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

