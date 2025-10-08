import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train["Age"] = df_train['Age'].fillna(-1)

print(df_train.describe())
df = pd.concat([df_train, df_test], sort=False, ignore_index=True)

df = df.rename(columns={'Fare': 'TicketFare'})

df["Fare"] = df["TicketFare"] / df.groupby("Ticket")["Ticket"].transform("count") # Netejar casos on tickets comprats en grup conten per cada membre

df = df.drop(columns=["Name", "Cabin"])
df = df.dropna()

# print(df.head())
# print(df.describe())

# for c in df.columns:
#     sns.displot(df[c])
#     plt.show()
    
# sns.pairplot(df)
# plt.show()
