import pandas as pd

data = pd.read_csv('fireendt.txt', delimiter = '~')
print(len(data))
