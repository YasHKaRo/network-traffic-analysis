import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class BinaryTrafficClassifier:
    """
    Класс для бинарной классификации сетевого трафика (Normal vs Attack)
    """
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None

    def initialize_models(self):
        """Инициализация моделей для сравнения"""
        print("Модели инициализированы")
        self.models = {
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            # Пока что не будем обучать, у меня нет времени
            #'SVM': SVC(
            #    probability=True,
            #    random_state=42,
            #    class_weight='balanced'
            #)

        }
        return self.models

    def train_models(self, X_train, y_train, X_test, y_test):
        """Обучение и оценка всех моделей"""
        self.results = {}

        for name, model in self.models.items():
            print(f"\n\tОбучение {name}...")

            # Обучение модели
            model.fit(X_train, y_train)

            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Метрики
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            # Сохраняем результаты
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            print(f"\t\t{name}: Accuracy = {round(accuracy, 4)}, AUC = {round(auc_score, 4)}")

        # Выбираем лучшую модель
        self._select_best_model()
        return self.results

    def _select_best_model(self):
        """Выбор лучшей модели по AUC score"""
        best_auc = 0
        for name, result in self.results.items():
            if result['auc_score'] > best_auc:
                best_auc = result['auc_score']
                self.best_model = result['model']
                self.best_model_name = name

        print(f"\n ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name} (AUC = {round(best_auc, 4)})")

    def get_feature_importance(self, feature_names, top_n=15):
        """Анализ важности признаков для лучшей модели"""
        if self.best_model_name in ['RandomForest', 'XGBoost', 'GradientBoosting']:
            if hasattr(self.best_model, 'feature_importances_'):
                importance = self.best_model.feature_importances_
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)

                print(f"\nТоп-{top_n} самых важных признаков:")
                print(self.feature_importance.head(top_n))

                return self.feature_importance.head(top_n)
        else:
            print("!! Для этой модели анализ важности признаков недоступен !!")
            return None

    def save_best_model(self, filepath):
        """Сохранение лучшей модели"""
        if self.best_model is not None:
            joblib.dump(self.best_model, filepath)
            print(f"\tЛучшая модель сохранена: {filepath}")
        else:
            print("!!Нет обученной модели для сохранения")


def evaluate_model_performance(y_true, y_pred, y_pred_proba, model_name):
    """Детальная оценка производительности модели"""
    print(f"\n{'=' * 50}")
    print(f"ДЕТАЛЬНАЯ ОЦЕНКА: {model_name}")
    print(f"{'=' * 50}")

    print("\nОтчет о классификации:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Матрица ошибок - {model_name}')
    plt.ylabel('Настоящее значение')
    plt.xlabel('Спрогнозированное значение')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {round(auc_score, 2)})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Частота ложно-положительных результатов')
    plt.ylabel('Частота истинно-положительных результатов')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

    return {
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': cm,
        'auc_score': auc_score
    }