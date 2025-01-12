import pandas as pd

students = {"hamed": 18,
            "ali": 12,
            "mousa": 17,
            "sara": 14.5,
            "emad": 19.25,
            "hossein": 18}

print(students)

ds = pd.Series(students)
print(ds)