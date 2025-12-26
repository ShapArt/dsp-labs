import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Загрузка и первичный обзор
data = pd.read_csv("train.csv")
print("=== info ===")
data.info()
print("=== head ===")
print(data.head())
print("=== describe ===")
print(data.describe())
print(f"=== shape: {data.shape} ===")

# Пропуски: заполнение медианой для числовых
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)
        print(f"filled NaN in '{col}' with median")


# Удаление выбросов (правило IQR)
def eliminate_outliers(df):
    for feature in df.select_dtypes(include=[np.number]).columns:
        if df[feature].nunique() == 2:
            continue
        Q1, Q3 = df[feature].quantile(0.25), df[feature].quantile(0.75)
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        before = df.shape[0]
        df = df[(df[feature] >= low) & (df[feature] <= high)]
        after = df.shape[0]
        if before != after:
            print(f"removed {before - after} outliers from '{feature}'")
    return df


# Удаление дубликатов
def drop_similar_rows(df):
    initial = df.shape[0]
    df = df.drop_duplicates()
    mask = df.duplicated(keep=False)
    df = df[~mask]
    final = df.shape[0]
    if initial != final:
        print(f"removed {initial - final} duplicates")
    return df


# Удаление неинформативных столбцов
def drop_irrelevant_columns(df):
    to_drop = ["id", "CustomerId", "Surname"]
    existing = [col for col in to_drop if col in df.columns]
    df = df.drop(columns=existing)
    if existing:
        print(f"dropped columns: {existing}")
    return df


processed = eliminate_outliers(data.copy())
processed = drop_similar_rows(processed)
processed = drop_irrelevant_columns(processed)

print("\n=== после очистки ===")
print(f"shape: {processed.shape}")
print(processed.dtypes)
print("NaN any:", processed.isnull().any().any())

# Кодирование категорий
if "Gender" in processed.columns:
    processed["Gender"] = processed["Gender"].map({"Male": 0, "Female": 1})
    print("Gender encoded")
if "Geography" in processed.columns:
    processed["Geography"] = processed["Geography"].map(
        {"France": 0, "Spain": 1, "Germany": 2}
    )
    print("Geography encoded")

# Целевой признак и средние по группам
target = "Exited"
if target not in processed.columns:
    raise ValueError(f"'{target}' not found")
print("\n=== grouped mean ===")
print(processed.groupby(target).mean())

# Визуализация
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", hue=target, data=processed)
plt.title("Gender vs Exited")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="Geography", hue=target, data=processed)
plt.title("Geography vs Exited")
plt.tight_layout()
plt.show()

sns.jointplot(x="Age", y="Balance", data=processed, kind="scatter")
plt.suptitle("Age vs Balance")
plt.show()

sns.jointplot(x="CreditScore", y="Balance", data=processed, kind="scatter")
plt.suptitle("CreditScore vs Balance")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(processed.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

print("\n=== crosstabs ===")
print(pd.crosstab(processed["Gender"], processed[target]))
print(pd.crosstab(processed["Geography"], processed[target]))

plt.figure(figsize=(8, 5))
sns.boxplot(x=target, y="Age", data=processed)
plt.title("Age vs Exited")
plt.tight_layout()
plt.show()
