from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r"Week3\petrol_consumption.csv")
x = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


regressor = LinearRegression()
regressor.fit(x_train, y_train)

sgd_regressor = SGDRegressor(max_iter=1000)
sgd_regressor.fit(x_train, y_train)


coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=["coefficient"])
sgd_coeff_df = pd.DataFrame(sgd_regressor.coef_, x.columns, columns=["coefficient"])

#print(coeff_df)
#print(sgd_coeff_df)


"""

y_pred = regressor.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)

print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAE: {metrics.root_mean_squared_error(y_test, y_pred)}")

"""
