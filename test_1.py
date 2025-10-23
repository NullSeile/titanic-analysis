import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# make it christmas
# COLOR = 'gold'
# mpl.rcParams['text.color'] = COLOR
# mpl.rcParams['axes.labelcolor'] = COLOR
# mpl.rcParams['xtick.color'] = COLOR
# mpl.rcParams['ytick.color'] = COLOR

def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y

def z_score_all_but_target(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    cols.remove("Survived")
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    return df

def z_score_norm(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()

def linear_norm(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())

def show_histograms(df: pd.DataFrame) -> None:
    for label in df.columns:
        plt.hist(df[label], bins=30)
        plt.title(f'Histogram of {label}')
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.show()

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df = pd.concat([
    df_train,
    df_test
], sort=False, ignore_index=True)

df = df.rename(columns={'Fare': 'TicketFare'})
df["Fare"] = df["TicketFare"] / df.groupby("Ticket")["Ticket"].transform("count") # Netejar casos on tickets comprats en grup conten per cada membre

df = df.dropna(subset=["Survived"])  # Eliminar files sense etiqueta de supervivència
df = df.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"])
# df = df.drop(columns=["Embarked"])  # Per ara, eliminar Embarked per simplicitat

# Count how many missing Age values there are
missing_age_count = df["Age"].isna().sum()
print(f"Missing Age values: {missing_age_count}, length of df: {len(df)}")

df["HasAge"] = df["Age"].notna().astype(int)
df["Age"] = df["Age"].fillna(df["Age"].median())
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

# print(df.head())
# print(df.describe())
# exit()

# Remove outliers
for col in ["Age", "TicketFare", "Fare"]:
    val = df[col].mean() + 3 * df[col].std()
    df.loc[df[col] > val, col] = val

# show_histograms(df)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    f1_score,
    classification_report,
    PrecisionRecallDisplay,
    roc_curve,
)

# from sklearn.model_selection import train_test_split

models = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Logistic Regression L2 1": lambda: LogisticRegression(max_iter=1000, penalty='l2', C=1),
    "Logistic Regression L2 0.1": lambda: LogisticRegression(max_iter=1000, penalty='l2', C=0.1),
    "Logistic Regression L2 0.01": lambda: LogisticRegression(max_iter=1000, penalty='l2', C=0.01),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, criterion='gini'),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "SVM linear": lambda: SVC(kernel='linear', C=1),
    "SVM rbf": lambda: SVC(kernel='rbf', C=1)
}

metrics = {
    "Accuracy ": accuracy_score,
    "Precision": average_precision_score,
    "Recall   ": recall_score,
    "F1 Score ": f1_score,
    # "Classification Report": classification_report
}

n = 100
model_metrics = {}
for model_name, make_model in models.items():
    train_scores = {metric: 0 for metric in metrics}
    val_scores = {metric: 0 for metric in metrics}

    for i in range(n):
            
        # Maybe do shuffle here
        # df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.dropna()

        # Standardize scores
        df = z_score_all_but_target(df)
        df = df.dropna()

        # Split train and validation sets (80% train, 20% validation)
        num_train = int(0.8 * len(df))
        df_train = df.iloc[:num_train]
        df_val = df.iloc[num_train:]

        X_train, y_train = split_X_y(df_train)
        X_val, y_val = split_X_y(df_val)

        # Standardize scores
        # X_train = z_score_norm(X_train)
        # X_val = z_score_norm(X_val)

        # model = LogisticRegression(max_iter=1000)
        model = make_model()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        for metric_name, metric_func in metrics.items():
            train_scores[metric_name] += metric_func(y_train, y_train_pred)
            val_scores[metric_name] += metric_func(y_val, y_val_pred)


        model_metrics[model_name] = (train_scores, val_scores)

    print(f"{model_name}")
    for metric_name in metrics:
        print(f"  {metric_name}: {train_scores[metric_name] / n:.5f} (Train), {val_scores[metric_name] / n:.5f} (Validation)")

    # PRECISIO-RECALL CURVE

    # pr_precision, pr_recall = precision_recall_curve(y_test, PREDICTIONS)
    # disp = PrecisionRecallDisplay(precision = pr_precision, recall = pr_recall)
    # disp.plot()
    # plt.show()
    
    # ROC CURVE
    
    # false_positives, true_positives = roc_curve(y_test, )
    

# Plot the metrics in a single plot without line
plt.figure(figsize=(12, 8))
for model_name, (train_scores, val_scores) in model_metrics.items():
    plt.plot(
        list(metrics.keys())[:-1],
        [val_scores[metric_name] / n for metric_name in list(metrics.keys())[:-1]],
        marker='o',
        # linestyle='None',
        label=model_name
    )
plt.title('Model Performance Comparison')
plt.legend()
plt.show()

# model_coef = pd.Series(model.coef_[0], index=X_train.columns)
# model_coef = model_coef.sort_values(ascending=False, key=abs)
# print("Model coefficients:")
# print(model_coef)

# Plot the absolute values of the coefficients, but color them red if negative and green if positive
# plt.figure(figsize=(10, 6))
# colors = model_coef.apply(lambda x: 'red' if x < 0 else 'green').tolist()
# model_coef.abs().plot(kind='bar', color=colors)
# plt.title('Absolute Values of Model Coefficients')
# plt.xlabel('Features')
# plt.ylabel('Absolute Coefficient Value')
# plt.tight_layout()
# plt.show()
