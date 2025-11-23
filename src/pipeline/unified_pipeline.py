import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class NetworkTrafficAnalyzer:
    """
    Единая система анализа сетевого трафика
    Объединяет все этапы: предобработка, классификация, анализ
    """
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(models_dir)
        self.categorical_preprocessor = None
        self.numerical_normalizer = None
        self.binary_classifier = None
        self.multiclass_classifier = None
        self.multiclass_label_encoder = None
        self.is_loaded = False

    def load_models(self):
        """Загрузка всех моделей и препроцессоров"""
        try:
            print(" Загрузка моделей...")

            # Загрузка препроцессоров
            self.categorical_preprocessor = joblib.load(self.models_dir / "categorical_preprocessor.joblib")
            self.numerical_normalizer = joblib.load(self.models_dir / "numerical_normalizer.joblib")

            # Загрузка классификаторов
            self.binary_classifier = joblib.load(self.models_dir / "best_binary_classifier.joblib")

            multiclass_package = joblib.load(self.models_dir / "tuned_multiclass_classifier.joblib")
            self.multiclass_classifier = multiclass_package['model']
            self.multiclass_label_encoder = multiclass_package['label_encoder']

            self.is_loaded = True
            print("   Все модели загружены успешно!")
            print(f"   - Бинарный классификатор: {type(self.binary_classifier).__name__}")
            print(f"   - Многоклассовый классификатор: {type(self.multiclass_classifier).__name__}")
            print(f"   - Доступные типы атак: {list(self.multiclass_label_encoder.classes_)}")

        except Exception as e:
            print(f" Ошибка загрузки моделей: {e}")
            raise

    def preprocess_new_data(self, raw_data):
        """
        Предобработка новых сырых данных
        Тот же пайплайн, что использовался при обучении
        """
        if not self.is_loaded:
            raise ValueError("Сначала загрузите модели!")

        # Создаем DataFrame (поддерживаем разные форматы ввода)
        if isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        elif isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        elif isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        else:
            raise ValueError("Неверный формат данных")

        # Удаляем целевые переменные если они есть
        df = df.drop(['label', 'attack_cat'], axis=1, errors='ignore')

        print(f"🔧 Предобработка {len(df)} записей...")

        # Применяем тот же пайплайн предобработки
        df_processed = self.categorical_preprocessor.transform(df)
        df_normalized = self.numerical_normalizer.transform(df_processed)

        return df_normalized

    def analyze_traffic(self, raw_data):
        """
        Полный анализ сетевого трафика
        Возвращает детальные результаты с объяснениями
        """
        # Предобработка
        processed_data = self.preprocess_new_data(raw_data)

        # Бинарная классификация
        binary_predictions = self.binary_classifier.predict(processed_data)
        binary_probabilities = self.binary_classifier.predict_proba(processed_data)

        # Многоклассовая классификация только для атак
        multiclass_results = ['Normal'] * len(processed_data)
        multiclass_confidences = [0.0] * len(processed_data)
        attack_details = [{}] * len(processed_data)

        attack_indices = np.where(binary_predictions == 1)[0]
        if len(attack_indices) > 0:
            X_attacks = processed_data[attack_indices]
            attack_type_predictions = self.multiclass_classifier.predict(X_attacks)
            attack_type_probabilities = self.multiclass_classifier.predict_proba(X_attacks)

            # Декодируем предсказания
            decoded_attacks = self.multiclass_label_encoder.inverse_transform(attack_type_predictions)

            for i, idx in enumerate(attack_indices):
                multiclass_results[idx] = decoded_attacks[i]
                multiclass_confidences[idx] = np.max(attack_type_probabilities[i])

                # Детали по всем типам атак
                attack_details[idx] = {
                    attack_type: float(prob) for attack_type, prob in zip(
                        self.multiclass_label_encoder.classes_,
                        attack_type_probabilities[i]
                    )
                }

        # Формируем финальные результаты
        results = []
        for i in range(len(processed_data)):
            is_attack = binary_predictions[i] == 1
            attack_confidence = binary_probabilities[i][1] if is_attack else binary_probabilities[i][0]

            result = {
                'record_id': i,
                'is_attack': bool(is_attack),
                'attack_type': multiclass_results[i],
                'confidence': float(attack_confidence),
                'attack_type_confidence': multiclass_confidences[i],
                'risk_level': self._assess_risk_level(multiclass_results[i], multiclass_confidences[i]),
                'recommended_action': self._get_recommended_action(multiclass_results[i], multiclass_confidences[i]),
                'detailed_probabilities': {
                    'normal': float(binary_probabilities[i][0]),
                    'attack': float(binary_probabilities[i][1]),
                    'attack_types': attack_details[i]
                }
            }
            results.append(result)

        return results

    def _assess_risk_level(self, attack_type, confidence):
        """Оценка уровня риска на основе типа атаки и уверенности"""
        high_risk_attacks = ['DoS', 'Exploits', 'Backdoor']
        medium_risk_attacks = ['Analysis', 'Reconnaissance', 'Shellcode']

        if attack_type == 'Normal':
            return 'low'

        if attack_type in high_risk_attacks and confidence > 0.7:
            return 'critical'
        elif attack_type in high_risk_attacks:
            return 'high'
        elif attack_type in medium_risk_attacks and confidence > 0.7:
            return 'high'
        else:
            return 'medium'

    def _get_recommended_action(self, attack_type, confidence):
        """Рекомендуемое действие на основе анализа"""
        if attack_type == 'Normal':
            return "Продолжить мониторинг"

        actions = {
            'critical': "НЕМЕДЛЕННОЕ БЛОКИРОВАНИЕ + УВЕДОМЛЕНИЕ АДМИНИСТРАТОРА",
            'high': "Блокирование источника + анализ логов",
            'medium': "Уведомление администратора + мониторинг",
            'low': "Запись в лог для последующего анализа"
        }

        risk_level = self._assess_risk_level(attack_type, confidence)
        return actions.get(risk_level, "Мониторинг")

    def generate_security_report(self, analysis_results):
        """Генерация сводного отчета о безопасности"""
        total_records = len(analysis_results)
        attacks = [r for r in analysis_results if r['is_attack']]

        if not attacks:
            return " Безопасно: атак не обнаружено"

        report = f" ОТЧЕТ О БЕЗОПАСНОСТИ\n{'=' * 40}\n"
        report += f"Всего проанализировано записей: {total_records}\n"
        report += f"Обнаружено атак: {len(attacks)}\n\n"

        # Группируем по типам атак
        attack_counts = {}
        for attack in attacks:
            attack_type = attack['attack_type']
            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

        report += "ТИПЫ ОБНАРУЖЕННЫХ АТАК:\n"
        for attack_type, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
            high_confidence_attacks = [a for a in attacks if
                                       a['attack_type'] == attack_type and a['attack_type_confidence'] > 0.7]
            report += f"- {attack_type}: {count} (высокая уверенность: {len(high_confidence_attacks)})\n"

        # Критические атаки
        critical_attacks = [a for a in attacks if a['risk_level'] == 'critical']
        if critical_attacks:
            report += f"\n КРИТИЧЕСКИЕ АТАКИ: {len(critical_attacks)}\n"
            for attack in critical_attacks[:3]:  # покажем первые 3
                report += f"  - {attack['attack_type']} (уверенность: {attack['attack_type_confidence']:.1%})\n"

        return report


# Пример использования системы
def demo_system():
    """Демонстрация работы полной системы"""
    analyzer = NetworkTrafficAnalyzer()
    analyzer.load_models()

    # Примеры сетевого трафика для тестирования
    sample_traffic = [
        {
            'proto': 'tcp', 'service': 'http', 'state': 'FIN',
            'dur': 0.5, 'sbytes': 560, 'dbytes': 480, 'sttl': 64, 'dttl': 58
        },
        {
            'proto': 'udp', 'service': 'dns', 'state': 'CON',
            'dur': 120.5, 'sbytes': 1500, 'dbytes': 1500, 'sttl': 128, 'dttl': 128
        }
    ]

    print("Анализ сетевого трафика:")

    results = analyzer.analyze_traffic(sample_traffic)

    for result in results:
        if result['is_attack']:
            print(f"   ЗАПИСЬ {result['record_id']}: АТАКА")
            print(f"   Тип: {result['attack_type']}")
            print(f"   Уверенность: {result['confidence']:.1%}")
            print(f"   Уровень риска: {result['risk_level'].upper()}")
            print(f"   Действие: {result['recommended_action']}")
        else:
            print(f"   ЗАПИСЬ {result['record_id']}: НОРМАЛЬНЫЙ ТРАФИК")
            print(f"   Уверенность: {result['confidence']:.1%}")
        print()

    # Сводный отчет
    report = analyzer.generate_security_report(results)
    print(report)

    return results


if __name__ == "__main__":
    demo_system()