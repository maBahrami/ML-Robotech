import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

m = regressor.coef_[0]
b = regressor.intercept_
print(f"y = {m}x + {b}")


y_pred = regressor.predict(x_test)
df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df)




