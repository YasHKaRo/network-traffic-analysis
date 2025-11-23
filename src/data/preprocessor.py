import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')
class CategoricalPreprocessor:
    """
    Класс для предобработки категориальных признаков UNSW-NB15
    """
    def __init__(self):
        self.categorical_features = ['proto', 'service', 'state']
        self.preprocessor = None
        self.is_fitted = False

    def create_preprocessing_pipeline(self):
        # Пайплайн для категориальных признаков
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # ColumnTransformer применяет разные преобразования к разным колонкам
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'  # остальные колонки оставляем как есть
        )

        return self.preprocessor

    def fit(self, df):
        """Обучение препроцессора на данных"""
        if self.preprocessor is None:
            self.create_preprocessing_pipeline()

        # Убедимся, что все категориальные признаки есть в данных
        missing_cols = [col for col in self.categorical_features if col not in df.columns]
        if missing_cols:
            print(f"Предупреждение: отсутствуют колонки {missing_cols}")

        self.preprocessor.fit(df)
        self.is_fitted = True
        print("Препроцессор обучен")

        # Сохраняем имена фичей после One-Hot Encoding
        self.feature_names = self.preprocessor.get_feature_names_out()
        return self

    def transform(self, df):
        """Применение препроцессора к данным"""
        if not self.is_fitted:
            raise ValueError("Сначала нужно обучить препроцессор методом .fit()")

        transformed_data = self.preprocessor.transform(df)

        # Преобразуем в DataFrame с правильными именами колонок
        df_transformed = pd.DataFrame(
            transformed_data,
            columns=self.feature_names,
            index=df.index
        )

        print(f"Данные преобразованы: {df_transformed.shape[1]} признаков")
        return df_transformed

    def fit_transform(self, df):
        """Обучение и преобразование за один шаг"""
        return self.fit(df).transform(df)

    def save(self, filepath):
        """Сохранение обученного препроцессора"""
        if not self.is_fitted:
            raise ValueError("Нечего сохранять - препроцессор не обучен")

        joblib.dump(self, filepath)
        print(f"Препроцессор сохранен в {filepath}")

    @classmethod
    def load(cls, filepath):
        """Загрузка обученного препроцессора"""
        preprocessor = joblib.load(filepath)
        print(f"Препроцессор загружен из {filepath}")
        return preprocessor


def analyze_categorical_features(df, categorical_columns):
    print("Анализ категориальных признаков")

    for col in categorical_columns:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(f"Уникальных значений: {df[col].nunique()}")
            print(f"Пропущенных значений: {df[col].isnull().sum()}")
            print("Топ-5 значений:")
            print(df[col].value_counts().head())

            # Рекомендации по обработке
            unique_count = df[col].nunique()
            if unique_count > 50:
                print(f"Много уникальных значений! Думай чё делать")
            elif unique_count <= 15:
                print(f"Подходит для One-Hot Encoding")
            else:
                print(f"Тяжело... Ну пускай будет One-Hot Encoding")


def handle_rare_categories(df, categorical_columns, threshold=0.01):
    """
    Обработка редких категорий - объединение в 'other'
    """
    df_processed = df.copy()

    for col in categorical_columns:
        if col in df.columns:
            # Вычисляем частоты
            value_counts = df[col].value_counts(normalize=True)
            # Находим редкие категории (меньше threshold)
            rare_categories = value_counts[value_counts < threshold].index
            # Заменяем редкие категории на 'other'
            df_processed[col] = df_processed[col].replace(rare_categories, 'other')

            if len(rare_categories) > 0:
                print(f"Объединено {len(rare_categories)} редких категорий в '{col}'")

    return df_processed