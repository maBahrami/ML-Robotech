import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import math


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

print(f"m: {(reg.coef_)[0]}")
print(f"b: {(reg.intercept_)}\n")

print(f"MAE: {round(mean_absolute_error(y_test, y_pred), 2)}")
print(f"MSE: {round(mean_squared_error(y_test, y_pred), 2)}")
print(f"RMSE: {round(root_mean_squared_error(y_test, y_pred), 2)}")



print("\n\n------------------ SGD Regressor ---------------\n")
SGDreg = SGDRegressor(max_iter=1000, random_state=123, eta0=0.001, shuffle=False)
SGDreg.fit(x_train, y_train)


y_pred = SGDreg.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)

print(f"\nepoch: {SGDreg.n_iter_}")
print(f"SGD update: {SGDreg.t_}\n")

print(f"m: {(SGDreg.coef_)[0]}")
print(f"b: {(SGDreg.intercept_)[0]}\n")

print(f"MAE: {round(mean_absolute_error(y_test, y_pred), 2)}")
print(f"MSE: {round(mean_squared_error(y_test, y_pred), 2)}")
print(f"RMSE: {round(root_mean_squared_error(y_test, y_pred), 2)}")

print("\n\n--------------- DIY SGD Regressor ---------------\n")
# yHat = mx + b
alpha = 0.001
m = 0
b = 0

step = 0
epoch = 0


y_train = y_train.to_numpy()
x_train = x_train[:, 0]

for _ in range(40):
    for i in range(0, len(x_train)):
        #print(m, b)

        yHat = m * x_train[i] + b
        yReal = y_train[i]

        Dm = -2 * x_train[i] * (yReal - yHat)
        Db = -2 * (yReal - yHat)

        m -= alpha * Dm
        b -= alpha * Db 

        step += 1
    epoch += 1
    #print("--------------------------------")

print(f"epoch: {epoch}")
print(f"SGD update: {step}")

print(f"\nm: {m}")
print(f"b: {b}")


y_test = y_test.to_numpy()
x_test = x_test[:, 0]

MAE = 0
MSE = 0
RMSE = 0
for i in range(len(x_test)):
    yHat = m * x_test[i] + b
    yReal = y_test[i]

    MAE += (abs(yHat - yReal)) / len(x_test)
    MSE += ((yHat - yReal)**2) / len(x_test)

print(f"\nMAE: {MAE}")
print(f"MSE: {MSE}")
print(f"RMSE: {math.sqrt(MSE)}\n\n")