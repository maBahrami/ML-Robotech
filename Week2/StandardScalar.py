from sklearn.preprocessing import StandardScaler

data = [[0, 0], 
        [0, 0], 
        [1, 1], 
        [1, 1]]

scaler = StandardScaler()

print(data)

scaler.fit(data)

print(scaler.mean_)

newData = scaler.transform(data)

print(newData)


new_data2 = scaler.fit_transform(data)
print(new_data2)


