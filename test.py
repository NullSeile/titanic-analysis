import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

COLOR = 'gold'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y

def z_score(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()

print(len(pd.read_csv("train.csv")))

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df = pd.concat([
    df_train,
    df_test
], sort=False, ignore_index=True)

df = df.rename(columns={'Fare': 'TicketFare'})
df["Fare"] = df["TicketFare"] / df.groupby("Ticket")["Ticket"].transform("count") # Netejar casos on tickets comprats en grup conten per cada membre

# df = df.dropna(subset=["Survived"])  # Eliminar files sense etiqueta de supervivència
df = df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])
# df = df.drop(columns=["Embarked"])  # Per ara, eliminar Embarked per simplicitat
df = df.dropna()

embarked = df["Embarked"]
df = df.drop(columns=["Embarked"])

sex = df["Sex"]
df = df.drop(columns=["Sex"])


df["Embarked_C"] = 0
df["Embarked_Q"] = 0
df["Embarked_S"] = 0
df.loc[embarked == 'C', "Embarked_C"] = 1
df.loc[embarked == 'Q', "Embarked_Q"] = 1
df.loc[embarked == 'S', "Embarked_S"] = 1

# Set sex from categorical to numerical [-1, 1]
df['Sex'] = sex.map({'female': 1, 'male': -1})

print(df.head())
print(df.describe())
# exit()

# Maybe do shuffle here
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)


# Split train and validation sets (80% train, 20% validation)
num_train = int(0.8 * len(df))
df_train = df.iloc[:num_train]
df_val = df.iloc[num_train:]

X_train, y_train = split_X_y(df_train)
X_val, y_val = split_X_y(df_val)

X_train = z_score(X_train)
X_val = z_score(X_val)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
val_accuracy = model.score(X_val, y_val)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

model_coef = pd.Series(model.coef_[0], index=X_train.columns)
model_coef = model_coef.sort_values(ascending=False, key=abs)
print("Model coefficients:")
print(model_coef)

# Plot the absolute values of the coefficients, but color them red if negative and green if positive
plt.figure(figsize=(10, 6))
colors = model_coef.apply(lambda x: 'red' if x < 0 else 'green').tolist()
model_coef.abs().plot(kind='bar', color=colors)
plt.title('Absolute Values of Model Coefficients')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()

