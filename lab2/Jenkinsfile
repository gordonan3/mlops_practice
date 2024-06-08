#!/bin/bash

# Устанавливаем флаг для остановки скрипта при ошибке
set -e

echo "Запуск data_creation.py"
python3 data_creation.py

echo "Запуск data_preprocessing.py"
python3 data_preprocessing.py

echo "Запуск model_preparation.py"
python3 model_preparation.py

echo "Запуск model_testing.py"
python3 model_testing.py

echo "Все скрипты успешно выполнены"
