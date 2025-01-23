import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv(r"Week3\HW\Datasets\Q3_data.csv")
#print(dataset)

dataset = dataset.loc[(dataset.iloc[:, 0])!=0]
#print(dataset)
#print(dataset.shape)


x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]
#print(x)
#print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print("\n\n--------------- Linear Regression -------------\n")
reg = LinearRegression()
reg.fit(x_train, y_train)

"""
m = round(reg.coef_[0], 2)
b = round(reg.intercept_, 2)
print(f"y = {m}x + {b}")
"""

y_pred = reg.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)

print(f"MAE: {round(mean_absolute_error(y_test, y_pred), 2)}")
print(f"MSE: {round(mean_squared_error(y_test, y_pred), 2)}")
print(f"RMSE: {round(root_mean_squared_error(y_test, y_pred), 2)}")



print("\n\n------------------ SGD Regressor ---------------\n")
SGDreg = SGDRegressor(max_iter=1000, random_state=123, eta0=0.01, shuffle=False)
SGDreg.fit(x_train, y_train)


y_pred = SGDreg.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)

print(f"MAE: {round(mean_absolute_error(y_test, y_pred), 2)}")
print(f"MSE: {round(mean_squared_error(y_test, y_pred), 2)}")
print(f"RMSE: {round(root_mean_squared_error(y_test, y_pred), 2)}")


print(f"\nepoch: {SGDreg.n_iter_}")
print(f"SGD update: {SGDreg.t_}")

print(SGDreg.coef_)
print(SGDreg.intercept_)


print("\n\n--------------- DIY SGD Regressor ---------------\n")
# yHat = mx + b
alpha = 0.01
m0 = 0
b0 = 0
margin = 0.001


y_train = y_train.to_numpy()
x_train = x_train[:, 0]


counter = 0
for _ in range(500):
    #print(m0 , b0)
    m = m0
    b = b0
    for i in range(0, len(x_train)):
        yHat = m * x_train[i] + b
        yReal = y_train[i]

        Dm = -2 * x_train[i] * (yReal - yHat)
        Db = -2 * (yReal - yHat)

        m0 -= alpha * Dm
        b0 -= alpha * Db

        counter += 1

print()
print(m)
print(b)

