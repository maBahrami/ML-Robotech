import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor


dataset = pd.read_csv(r"Week7\HW\reference\Q2\house_price.csv")
x = dataset.iloc[:, 0:2]
y = dataset.iloc[:, 2]

maxPrice = y.max()
y = y / maxPrice


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



# ------------------------------ Neural Network ------------------------------
EPOCHS = 50
BATCH_SIZE = 2

net = models.Sequential([
                            layers.Dense(32, activation="relu"),
                            layers.Dense(8, activation="relu"),
                            layers.Dense(1, activation="linear")
                        ])

net.compile(optimizer="adam",
            loss="mse")

H = net.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

loss = net.evaluate(x_test, y_test)
print(f"loss: {loss}")

y_pred = net.predict(x_test)
y_pred.flatten()


print("\n\n--------------------- Neural Network -------------------")

print(f"\nMAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_pred)}\n")


plt.style.use("ggplot")
plt.plot(H.history["loss"], label = "train loss")
plt.plot(H.history["val_loss"], label = "test loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title(f"batch size: {BATCH_SIZE}, epochs: {EPOCHS}")
plt.legend()
plt.show()



sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print("\n--------------------- Linear Regression -------------------")
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#print(regressor.coef_)

y_pred = regressor.predict(x_test)
print(f"\nMAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_pred)}\n")

#df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)


print("\n--------------------- SGD Regressor -------------------")
sgdReg = SGDRegressor(random_state=123, eta0=0.01)
sgdReg.fit(x_train, y_train)

#print(sgdReg.coef_)

y_pred = sgdReg.predict(x_test)
print(f"\nMAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAPE: {metrics.mean_absolute_percentage_error(y_test, y_pred)}\n")
# pay attetion to MAPE

#df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)

#epoch = sgdReg.n_iter_
#print(f"epoch: {epoch}")

#GD_update = sgdReg.t_
#print(f"GD updates: {GD_update}")