import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Titanic-Dataset.csv')
print("Dataset Info")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])


df=df.drop('Cabin', axis=1)


print(df.isnull().sum())


df = pd.get_dummies(df, columns=['Sex', 'Embarked'])


print(df.head())


