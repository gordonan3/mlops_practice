import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Стандартизация числовых признаков
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Папки с данными
train_folder = 'train'
test_folder = 'test'

i = 0
# Считываем и обрабатываем данные из папки train
for file in os.listdir(train_folder):
    if file.startswith("train_features"):
        train_features = pd.read_csv(os.path.join(train_folder, file))
        X_train_scaled = scaler.fit_transform(train_features)
        X_train_scaled_df = pd.DataFrame(X_train_scaled)
        X_train_scaled_df.to_csv(f'train_scaled/X_train_scaled_{i}.csv', index=False)
        i += 1

i = 0
for file in os.listdir(train_folder):
    if file.startswith("train_target"):
        train_target = pd.read_csv(os.path.join(train_folder, file))
        train_target_cod = label_encoder.fit_transform(train_target.values.ravel())
        train_target_cod_df = pd.DataFrame(train_target_cod)
        train_target_cod_df.to_csv(f'train_scaled/train_target_cod_{i}.csv', index=False)
        i += 1

# Сохранение обученного scaler и label encoder
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

i = 0
# Считываем и обрабатываем данные из папки test
for file in os.listdir(test_folder):
    if file.startswith("test_features"):
        test_features = pd.read_csv(os.path.join(test_folder, file))
        X_test_scaled = scaler.transform(test_features)
        X_test_scaled_df = pd.DataFrame(X_test_scaled)
        X_test_scaled_df.to_csv(f'test_scaled/X_test_scaled_{i}.csv', index=False)
        i +=1

i = 0
for file in os.listdir(test_folder):
    if file.startswith("test_target"):
        test_target = pd.read_csv(os.path.join(test_folder, file))
        test_target_cod = label_encoder.fit_transform(test_target.values.ravel())
        test_target_cod_df = pd.DataFrame(test_target_cod)
        test_target_cod_df.to_csv(f'test_scaled/test_target_cod_{i}.csv', index=False)
        i += 1

