import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Загрузка набора данных из директории скрипта
directoriya_skript = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(directoriya_skript, 'loan_approval_dataset.csv')
data = pd.read_csv(csv_path)
# Удаляем пробелы в названиях столбцов
data.columns = data.columns.str.strip()

# 2. Определение признаков (X) и целевой переменной (y)
X = data.drop(['loan_id', 'loan_status'], axis=1)
y = data['loan_status']

# 3. Разделение признаков на числовые и категориальные
numeric_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

# 4. Настройка препроцессинга
numeric_transformer = StandardScaler()  # Масштабирование числовых признаков
# Параметр sparse_output для кодирования категориальных признаков
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# 5. Разбиение данных на обучающую и тестовую выборки (25% тест)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=123
)

# 6. Описание моделей и сеток гиперпараметров
models_config = {
    'Логистическая регрессия': {
        'pipeline': Pipeline([
            ('preproc', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, random_state=123))
        ]),
        'params': {
            'clf__C': [0.01, 0.1, 1, 10],  # Коэффициент регуляризации
            'clf__penalty': ['l2']         # Тип регуляризации
        }
    },
    'Гауссовский наивный Байес': {
        'pipeline': Pipeline([
            ('preproc', preprocessor),
            ('clf', GaussianNB())
        ]),
        'params': {
            'clf__var_smoothing': [1e-9, 1e-8, 1e-7]  # Параметр сглаживания дисперсии
        }
    },
    'Случайный лес': {
        'pipeline': Pipeline([
            ('preproc', preprocessor),
            ('clf', RandomForestClassifier(random_state=123))
        ]),
        'params': {
            'clf__n_estimators': [50, 100, 200],  # Количество деревьев
            'clf__max_depth': [None, 10, 20]       # Максимальная глубина дерева
        }
    }
}

# 7. Поиск лучших гиперпараметров, оценка моделей и визуализация результатов
for name, config in models_config.items():
    print(f"\nМодель: {name}")
    grid = GridSearchCV(
        estimator=config['pipeline'],
        param_grid=config['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # Вывод лучших параметров
    print("Лучшие параметры:", grid.best_params_)

    # Предсказания на тестовой выборке
    y_pred = grid.predict(X_test)

    # Вычисление матрицы неточностей и отчета о классификации
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print("Матрица неточностей:\n", cm)
    print("Отчет о классификации:\n", cr)

    # Визуализация матрицы неточностей
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.title(f"Матрица неточностей: {name}")
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()
