from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r"Week3\HW\Datasets\Q2_house_price.csv")

#print(dataset.head())
x = dataset.iloc[:, 0:2]
y = dataset.iloc[:, 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


print("\n\n--------------------- Linear Regression -------------------\n")
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(regressor.coef_)

y_pred = regressor.predict(x_test)
print(f"\nMAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_pred)}\n")

df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df)


print("\n\n--------------------- SGD Regressor -------------------\n")
sgdReg = SGDRegressor(random_state=123, eta0=0.01)
sgdReg.fit(x_train, y_train)

print(sgdReg.coef_)

y_pred = sgdReg.predict(x_test)
print(f"\nMAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_pred)}\n")

df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df)

epoch = sgdReg.n_iter_
print(f"epoch: {epoch}")

GD_update = sgdReg.t_
print(f"GD updates: {GD_update}")