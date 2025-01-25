from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])

print(lb.classes_)

print(lb.transform([1, 6]))
