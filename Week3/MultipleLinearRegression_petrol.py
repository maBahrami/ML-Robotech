from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r"Week3\petrol_consumption.csv")
x = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=["coefficient"])
print(coeff_df)
print(f"intercept: \t\t\t{regressor.intercept_}\n")

y_pred = regressor.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
#print(df)

print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAE: {metrics.root_mean_squared_error(y_test, y_pred)}")

