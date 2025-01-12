import pandas as pd

# ______________________________________________________
#                     * Series *     
# ______________________________________________________

students = {"hamed": 18,
            "ali": 12,
            "mousa": 17,
            "sara": 14.5,
            "emad": 19.25,
            "hossein": 18}
#print(students)

ds = pd.Series(students)
'''
print(ds)

# -------------------- keys and values of the dictionary ------------------
print(list(students.keys()))
print(list(students.values()))



# -------------------- index and values of the Series ---------------------
print(list(ds.index))
print(ds.values)

# -------------------- element accessing in Series---------------------------------
print(ds["hamed"])

print(ds.ali)

print(ds[-1])

print(ds[["ali", "emad"]])

print([[0, 4]])

print(ds[:3])

print(ds["mousa" : "emad"])

print(ds[[True, False, False, True, True, False]])


# ------------------- math operation in Series --------------------------
# when we use the data in dictionay mode
marks = list(students.values())
avg = sum(marks) / len(marks)
for i in students:
    if students[i] > avg:
        print(i)
# when we use the data in Series data structure
print(ds[ds > ds.mean()].index)
'''


# ______________________________________________________
#                     * Data Frame *     
# ______________________________________________________

students = {"name": ["majid", "ali", "reza", "sara", "hasan", "mohammad", "ahmad", "maryam", "mina", "sahar"],
            "grade": [12, 5, 18, 16.5, 7, 17, 19, 13, 6, 8],
            "gender": ["M", "M", "M", "F", "M", "M", "M", "F", "F", "F"],
            "guest": [True, True, False, False, False, False, False, True, True, False],
            "city": ["Tehran", "Tehran", "Mashhad", "Yazd", "Shiraz", "Tehran", "Mashhad", "Yazd", "Karaj", "Tehran"]}

#print(students)

df = pd.DataFrame(students)

#print(df)

#rows, cols = df.shape
#print(f"nrows (samples): {rows}, ncols (features): {cols}.")

'''
print(df.columns)

print(df.index)
print(df.values)

print(df.head(3))
print(df.tail(3))
print(df.info())

print(df.name)

print(df[["city", "name"]])

print(df["grade"] > 10)


# ------------------------- iloc method ----------------------------------
# df.iloc[rows_number, cols_number]

print(df.iloc[0, 0])

print(df.iloc[3:, :2])

print(df.iloc[[0, 1], -3:])

print(df.iloc[[True, True, False, False, False, False, False, False, False, True], [0, 1]])


# ------------------------- loc method ----------------------------------
# df.loc[rows_label, cols_label]

print(df.loc[:, ["city", "gender"]])

print(df.loc[1:3, "grade":"guest"])

print(df.loc[[0, 1], ["gender", "city"]])


print(df.loc[[True, True, False, False, False, False, False, False, False, True], "name"])

'''


# ------------------------ CSV ------------------------------------------

df = pd.read_csv(r".......", sep = "\t", header=None, names=[........],
                 skiprows=2, comment="#",
                 usecols=["....", "....."])





