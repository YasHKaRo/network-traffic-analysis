import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class MulticlassTrafficClassifier:
    """
    Класс для многоклассовой классификации типов сетевых атак
    """
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.attack_types = None

    def initialize_models(self):
        """Инициализация моделей для многоклассовой классификации"""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                multi_class='ovr'
            )
        }
        print("Многоклассовые модели инициализированы")
        return self.models

    def prepare_labels(self, y_multiclass):
        """Подготовка и кодирование меток атак"""
        # Сохраняем оригинальные названия атак
        self.attack_types = y_multiclass.unique()
        print(f"Обнаружено типов атак: {len(self.attack_types)}")
        print("Типы атак:", list(self.attack_types))
        # Кодируем метки
        y_encoded = self.label_encoder.fit_transform(y_multiclass)
        return y_encoded

    def train_models(self, X_train, y_train, X_test, y_test):
        """Обучение и оценка всех многоклассовых моделей"""
        self.results = {}
        # Кодируем метки для тестовой выборки
        y_test_encoded = self.label_encoder.transform(y_test)
        for name, model in self.models.items():
            print(f"\n\tОбучение {name}...")
            try:
                # Обучение модели
                model.fit(X_train, y_train)

                # Предсказания
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                # Метрики
                accuracy = accuracy_score(y_test_encoded, y_pred)

                # Classification report
                report = classification_report(y_test_encoded, y_pred,
                                               target_names=self.label_encoder.classes_,
                                               output_dict=True)

                # Сохраняем результаты
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'classification_report': report
                }
                print(f"\t{name}: Accuracy = {accuracy:.4f}")
                print(f"\tПодробный отчет сохранен")
            except Exception as e:
                print(f"\tОшибка при обучении {name}: {e}")

        # Выбираем лучшую модель
        self._select_best_model()
        return self.results

    def _select_best_model(self):
        """Выбор лучшей модели по accuracy"""
        best_accuracy = 0
        for name, result in self.results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                self.best_model = result['model']
                self.best_model_name = name

        print(f"\n Лучшая многоклассовая модель: {self.best_model_name} (Accuracy = {best_accuracy:.4f})")

    def analyze_class_performance(self, y_true, y_pred, model_name):
        """Детальный анализ производительности по классам"""
        y_true_decoded = self.label_encoder.inverse_transform(y_true)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        print(f"\nАнализ по классам: {model_name}")

        # Classification Report
        print("\nОтчет о классификации:")
        print(classification_report(y_true_decoded, y_pred_decoded))

        # Confusion Matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true_decoded, y_pred_decoded,
                              labels=self.label_encoder.classes_)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Матрица ошибок - {model_name}\n(Многоклассовая классификация)')
        plt.ylabel('Истинный тип атаки')
        plt.xlabel('Предсказанный тип атаки')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        return cm

    def get_feature_importance(self, feature_names, top_n=15):
        """Анализ важности признаков для лучшей модели"""
        if self.best_model_name in ['RandomForest', 'XGBoost', 'RandomForest_OVR']:
            model_to_check = self.best_model
            if self.best_model_name == 'RandomForest_OVR':
                model_to_check = self.best_model.estimator

            if hasattr(model_to_check, 'feature_importances_'):
                importance = model_to_check.feature_importances_
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)

                print(f"\nТоп-{top_n} самых важных признаков для многоклассовой классификации:")
                print(self.feature_importance.head(top_n))

                return self.feature_importance.head(top_n)
        else:
            print("Для этой модели анализ важности признаков недоступен")
            return None

    def save_best_model(self, filepath):
        """Сохранение лучшей модели и label encoder"""
        if self.best_model is not None:
            # Сохраняем модель и label encoder вместе
            model_package = {
                'model': self.best_model,
                'label_encoder': self.label_encoder,
                'feature_names': getattr(self, 'feature_names', None)
            }
            joblib.dump(model_package, filepath)
            print(f"Лучшая многоклассовая модель сохранена: {filepath}")
        else:
            print("Нет обученной модели для сохранения")


def compare_multiclass_models(results):
    """Сравнение всех многоклассовых моделей"""
    comparison_data = []

    for name, result in results.items():
        report = result['classification_report']

        # Macro average metrics
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']

        comparison_data.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Macro Precision': macro_precision,
            'Macro Recall': macro_recall,
            'Macro F1': macro_f1
        })

    comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)

    print("\nСравнение многоклассовых моделей:")
    print(comparison_df)

    # Визуализация сравнения
    metrics_to_plot = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        sns.barplot(data=comparison_df, x=metric, y='Model', ax=ax, palette='viridis')
        ax.set_title(f'{metric} - Многоклассовая классификация')
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

    return comparison_df