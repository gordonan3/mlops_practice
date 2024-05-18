import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Создаем списки для хранения объединенных данных
X_train_scaled_combined = []
train_target_cod_combined = []

# Обрабатываем и объединяем данные из всех пяти наборов
for i in range(5):
    # Загружаем и обрабатываем данные из папок train_scaled и test_scaled
    X_train_scaled = pd.read_csv(f'train_scaled/X_train_scaled_{i}.csv')
    train_target_cod = pd.read_csv(f'train_scaled/train_target_cod_{i}.csv', header=None)

    # Добавляем данные в списки
    X_train_scaled_combined.append(X_train_scaled)
    train_target_cod_combined.append(train_target_cod.drop(index=0))

# Объединяем данные из списков в один DataFrame
X_train_scaled_combined = pd.concat(X_train_scaled_combined, ignore_index=True)
train_target_cod_combined = pd.concat(train_target_cod_combined, ignore_index=True)

# Создаем модель Gradient Boosting Classifier
model_rf = RandomForestClassifier(n_estimators=150, max_depth=10, oob_score=True)

# Обучаем модель
model_rf.fit(X_train_scaled_combined, train_target_cod_combined)

# Сохраняем обученную модель
joblib.dump(model_rf, 'trained_model.joblib')