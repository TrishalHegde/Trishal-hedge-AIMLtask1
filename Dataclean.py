import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Titanic-Dataset.csv')
print("Dataset Info")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

