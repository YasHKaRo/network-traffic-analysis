import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path


class NumericalNormalizer:
    """
    Класс для нормализации числовых признаков сетевого трафика
    """
    def __init__(self, method='standard'):
        """
        Args:
            method: 'standard' (StandardScaler) или 'minmax' (MinMaxScaler)
        """
        self.method = method
        self.numeric_columns = None
        self.normalizer = None
        self.is_fitted = False

    def identify_numeric_columns(self, df):
        """Автоматически идентифицирует числовые признаки"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Исключаем бинарные признаки (0/1) которые уже нормализованы
        binary_cols = []
        non_binary_numeric = []

        for col in numeric_cols:
            unique_vals = df[col].unique()
            if set(unique_vals).issubset({0, 1}) and len(unique_vals) <= 2:
                binary_cols.append(col)
            else:
                non_binary_numeric.append(col)

        print(f"Обнаружено:")
        print(f"\t- Числовых признаков для нормализации: {len(non_binary_numeric)}")
        print(f"\t- Бинарных признаков (пропущено): {len(binary_cols)}")

        return non_binary_numeric

    def create_normalization_pipeline(self, numeric_columns):
        """Создает пайплайн для нормализации"""
        if self.method == 'standard':
            scaler = StandardScaler()
        elif self.method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Метод должен быть 'standard' или 'minmax'")

        self.normalizer = ColumnTransformer(
            transformers=[
                ('num', scaler, numeric_columns)
            ],
            remainder='passthrough'  # остальные колонки (one-hot) не трогаем
        )

        return self.normalizer

    def fit(self, df):
        """Обучение нормализатора на данных"""
        if self.numeric_columns is None:
            self.numeric_columns = self.identify_numeric_columns(df)

        if self.normalizer is None:
            self.create_normalization_pipeline(self.numeric_columns)

        self.normalizer.fit(df)
        self.is_fitted = True

        print(f"\tНормализатор обучен (метод: {self.method})")
        print(f"\tНормализуется {len(self.numeric_columns)} числовых признаков")

        return self

    def transform(self, df):
        """Применение нормализатора к данным"""
        if not self.is_fitted:
            raise ValueError("Сначала нужно обучить нормализатор методом .fit()")

        transformed_data = self.normalizer.transform(df)

        # Сохраняем имена колонок
        feature_names = self.normalizer.get_feature_names_out()

        # Создаем DataFrame с правильными именами колонок
        df_transformed = pd.DataFrame(
            transformed_data,
            columns=feature_names,
            index=df.index
        )

        print(f"\tДанные нормализованы: {df_transformed.shape}")
        return df_transformed

    def fit_transform(self, df):
        """Обучение и преобразование за один шаг"""
        return self.fit(df).transform(df)

    def save(self, filepath):
        """Сохранение обученного нормализатора"""
        if not self.is_fitted:
            raise ValueError("Нечего сохранять - нормализатор не обучен")

        joblib.dump(self, filepath)
        print(f"\tНормализатор сохранен в {filepath}")

    @classmethod
    def load(cls, filepath):
        """Загрузка обученного нормализатора"""
        normalizer = joblib.load(filepath)
        print(f"\tНормализатор загружен из {filepath}")
        return normalizer


def analyze_numeric_features_before_normalization(df, numeric_columns):
    """Анализ числовых признаков перед нормализацией"""
    print("\n\tПредварительный анализ числовых признаков")

    stats = df[numeric_columns].describe().T
    stats['range'] = stats['max'] - stats['min']
    stats['cv'] = stats['std'] / stats['mean']  # коэффициент вариации

    print("\nСтатистики числовых признаков:")
    print(stats[['mean', 'std', 'min', 'max', 'range', 'cv']].round(3))

    # Рекомендации по нормализации
    high_variance = stats[stats['cv'] > 1.0].index.tolist()
    if high_variance:
        print(f"\nВысокая вариативность (CV > 1): {high_variance}")
        print("\tРекомендуется StandardScaler")

    return stats