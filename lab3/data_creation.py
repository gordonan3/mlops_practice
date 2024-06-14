import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

# Список месяцев года
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

def create_dataset(n_samples):

    # Создаем списки для температур и месяцев
    temperatures = []
    months_data = []
    humidity =[]

    def sin(x):

      f = np.sin(x)

      return f

    for i, month in enumerate(months):
        # Генерируем температуры для каждого месяца года в реальных диапазонах
        avg_temp = 15*sin(0.5*(i-3.3))     # Средняя температура
        avg_air = 18*sin(0.5*(i+5))+70    # Средняя относительная владность воздуха
        temp = np.random.uniform(low=avg_temp - 5, high=avg_temp + 5, size=n_samples)
        air = np.random.uniform(low=avg_air - 4, high=avg_air + 4, size=n_samples)
        # Добавляем температуры и соответствующие месяцы в списки
        temperatures.extend(temp)
        humidity.extend(air)
        months_data.extend([month] * n_samples)
    # Создаем DataFrame для признаков (температур)
    features_df = pd.DataFrame(list(zip(temperatures, humidity)))
    # Создаем DataFrame для целевой переменной (месяцев)
    target_df = pd.DataFrame(months_data)
    return features_df, target_df

# Создаем несколько наборов данных и сохраняем их в папки "train" и "test"
for i in range(5):
    X, y = create_dataset(500)

    # разбиваем на тестовую и валидационную
    train_features, test_features, train_target, test_target = train_test_split(X, y, test_size=0.3, random_state=42) 

    train_features.to_csv(f'train/train_features_{i}.csv', index=False)
    train_target.to_csv(f'train/train_target_{i}.csv', index=False)
    test_features.to_csv(f'test/test_features_{i}.csv', index=False)
    test_target.to_csv(f'test/test_target_{i}.csv', index=False)