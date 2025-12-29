import pandas as pd
import numpy as np

df = pd.DataFrame({
    'Age': [25, None, None, 35, -5, 28],
    'Salary': [50000, None, -70000, 80000, 45000, 50000],
    'Department': ['IT', None, 'HR', None, 'HR', 'IT']
})

median = df['Age'].median()

# df['Age'] = df['Age'].mask(df['Age'] == 0, median)
# df['Age'] = df['Age'].mask(df['Age'] < 0, 0)
# df['Age'] = df['Age'].where(df['Age'] >= 0, 0)
# df['Age'] = df['Age'].clip(lower=0, upper=40)
# df['Age'] = df['Age'].apply(lambda x: 0 if x < 0 else x)

# one way of cleaning NaNs is dropping them entirely
list1 = [("map1",2), ("map2",4), ("map3",6)]

dict1 = dict(list1)
# print(dict1)
datafr = pd.DataFrame([dict1])
print(datafr)



