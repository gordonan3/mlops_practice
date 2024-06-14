import os
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Загрузка обученной модели
model_rf = joblib.load('trained_model.joblib')

# Создаем списки для хранения объединенных данных тестовых наборов
X_test_scaled_combined = []
test_target_cod_combined = []


# Обрабатываем и объединяем данные из всех пяти тестовых наборов
for i in range(5):
    # Загружаем и обрабатываем данные из папки test_scaled
    X_test_scaled = pd.read_csv(os.path.join('test_scaled', f'X_test_scaled_{i}.csv'))
    test_target_cod = pd.read_csv(os.path.join('test_scaled', f'test_target_cod_{i}.csv'), header=None)

    # Добавляем данные в списки
    X_test_scaled_combined.append(X_test_scaled)
    test_target_cod_combined.append(test_target_cod.drop(index=0))

# Объединяем данные из списков в один DataFrame
X_test_scaled_combined = pd.concat(X_test_scaled_combined, ignore_index=True)
test_target_cod_combined = pd.concat(test_target_cod_combined, ignore_index=True)

# Выполняем предсказания с использованием обученной модели
predictions = model_rf.predict(X_test_scaled_combined)

# Вычисляем метрику точности
accuracy = accuracy_score(test_target_cod_combined, predictions)
print("Accuracy:", accuracy)