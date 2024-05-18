#!/bin/bash

# Устанавливаем флаг для остановки скрипта при ошибке
set -e

echo "Запуск data_creation.py"
python data_creation.py

echo "Запуск data_preprocessing.py"
python data_preprocessing.py

echo "Запуск model_preparation.py"
python model_preparation.py

echo "Запуск model_testing.py"
python model_testing.py

echo "Все скрипты успешно выполнены"