import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"Week2\HW\Datasets\Q3_population.csv")
#print(dataset)

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 1]
#print(x)
#print(y)

x.loc[:, 'population'] = (x.loc[:, 'population']).replace(0, np.nan)
mean = int((x.loc[:, 'population']).mean(skipna=True))
x.loc[:, 'population'] = (x.loc[:, 'population']).replace(np.nan, mean)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)


reg = LinearRegression()
reg.fit(x_train, y_train)

m = round(reg.coef_[0], 2)
b = round(reg.intercept_, 2)
print(f"y = {m}x + {b}")

y_pred = reg.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)

print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")


plt.scatter(x, y)
plt.xlabel("population")
plt.ylabel("benefit")
plt.show()


