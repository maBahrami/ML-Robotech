from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# -------------------- Ordinal ---------------------------
data = asarray([['second'], ['first'], ['third']])
print(data)

ordinal = OrdinalEncoder()
out = ordinal.fit_transform(data)

print(out)


# -------------------- Nominal --------------------------
data2 = asarray([['red'], ['green'], ['blue']])
print(data2)

encoder = OneHotEncoder(sparse_output=False)
out = encoder.fit_transform(data2)
print(out)