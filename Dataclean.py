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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['Age', 'Fare']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print(df[numerical_features].head())


sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()


Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

print(f"Shape of dataframe after removing outliers: {df.shape}")


