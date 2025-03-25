import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

import mapping

# Обучаем модели с использованием всех доступных ядер для тех, которые поддерживают n_jobs.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
gb_model = HistGradientBoostingRegressor(max_iter=100, random_state=42, loss='least_squares')
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)


# ======================
# 1. Загрузка и предобработка данных
# ======================
def load_data(csv_file='russia_real_estate_2021.csv'):
    """
    Загружает датасет
    """
    df = pd.read_csv(csv_file, delimiter=";")
    return df


def preprocess_data(df):
    """
    Предобработка данных:
    - Удаляются столбцы: date, postal_code, street_id, id_region, house_id
    - Удаляются строки с пропусками в ключевых признаках
    - Приведение типов для числовых признаков
    - Удаление выбросов по цене и площади
    """
    # Удаляем ненужные столбцы
    cols_to_drop = ["date", "postal_code", "street_id", "house_id"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Удаляем строки с пропусками в ключевых столбцах
    df = df.dropna(subset=["price", "area", "rooms", "level", "levels", "kitchen_area", "geo_lat", "geo_lon", 'building_type', 'object_type'])

    # Кодируем категориальные признаки
    df = encode_categorical_features(df)

    # Приводим столбцы к числовому типу
    numeric_cols = ["price", "area", "rooms", "level", "levels", "kitchen_area", "geo_lat", "geo_lon"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Удаление выбросов по цене
    Q1 = df["price"].quantile(0.25)
    Q3 = df["price"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["price"] >= Q1 - 1.5 * IQR) & (df["price"] <= Q3 + 1.5 * IQR)]

    # Удаление выбросов по площади
    Q1_area = df["area"].quantile(0.25)
    Q3_area = df["area"].quantile(0.75)
    IQR_area = Q3_area - Q1_area
    df = df[(df["area"] >= Q1_area - 1.5 * IQR_area) & (df["area"] <= Q3_area + 1.5 * IQR_area)]

    return df

# ======================
# 1.1 Hot end encoding
# ======================
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    building_type_OHE = mapping.building_type_encoder.fit_transform(df[['building_type']])
    building_type_df = pd.DataFrame(building_type_OHE, columns=mapping.building_type_column_names)

    object_type_OHE = mapping.object_type_encoder.fit_transform(df[['object_type']])
    object_type_df = pd.DataFrame(object_type_OHE, columns=mapping.object_type_column_names)

    df = (pd.concat([df, building_type_df, object_type_df], axis=1).
          drop('building_type', axis=1).
          drop('object_type', axis=1))

    return df

# ======================
# 2. Обучение моделей
# ======================
def train_models(df):
    """
    Обучение нескольких моделей (RandomForest, Gradient Boosting, XGBoost) с использованием признаков:
    rooms, area, kitchen_area, level, levels, building_type, object_type, geo_lat, geo_lon.
    Сохраняет обученные модели и экспортирует метрики.
    """
    features = list(filter(lambda x: x != "price",df.columns.to_list()))
    target = "price"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

    logging.info("RandomForestRegressor was started")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
    rf_model.fit(X_train, y_train)

    logging.info("GradientBoostingRegressor was started")
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=2)
    gb_model.fit(X_train, y_train)

    logging.info("XGBRegressor was started")
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=2)
    xgb_model.fit(X_train, y_train)

    models = {
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model
    }

    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae_val = mean_absolute_error(y_test, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_val = r2_score(y_test, y_pred)
        metrics.append({
            "Модель": name,
            "MAE": round(mae_val, 2),
            "RMSE": round(rmse_val, 2),
            "R2": round(r2_val, 3)
        })
        print(f"{name}: MAE = {mae_val:.2f}, RMSE = {rmse_val:.2f}, R2 = {r2_val:.3f}")

    # Сохраняем модели
    joblib.dump(rf_model, "model_rf.pkl")
    joblib.dump(gb_model, "model_gb.pkl")
    joblib.dump(xgb_model, "model_xgb.pkl")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("res/model_metrics_comparison.csv", index=False)
    print("Метрики моделей сохранены в 'model_metrics_comparison.csv'")

    corr_matrix = df.corr()
    return models, features, X_test, y_test, corr_matrix


# ======================
# 3. Визуализация данных и результатов
# ======================
def plot_visualizations(df, corr_matrix, models, X_test, y_test):
    # Тепловая карта корреляций
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Матрица корреляций признаков")
    plt.tight_layout()
    plt.savefig("res/correlation_heatmap.png")
    plt.close()

    # Гистограмма распределения цен
    plt.figure(figsize=(6, 4))
    sns.histplot(df["price"], kde=True, bins=50)
    plt.title("Распределение цен недвижимости")
    plt.xlabel("Цена")
    plt.ylabel("Количество объектов")
    plt.tight_layout()
    plt.savefig("res/price_distribution.png")
    plt.close()

    # Сравнение RMSE для моделей
    model_names = list(models.keys())
    rmse_values = []
    for model in models.values():
        y_pred = model.predict(X_test)
        rmse_val = mean_squared_error(y_test, y_pred)
        rmse_values.append(rmse_val)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=model_names, y=rmse_values)
    plt.title("Сравнение RMSE моделей")
    plt.ylabel("RMSE (ниже лучше)")
    plt.tight_layout()
    plt.savefig("res/model_rmse_comparison.png")
    plt.close()

    # Фактические vs Предсказанные цены для Random Forest
    best_model = models["Random Forest"]
    y_pred_best = best_model.predict(X_test)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred_best, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Фактическая цена")
    plt.ylabel("Предсказанная цена")
    plt.title("Фактическая vs Предсказанная цена (Random Forest)")
    plt.tight_layout()
    plt.savefig("res/actual_vs_predicted_rf.png")
    plt.close()

    print("Графики сохранены как PNG-файлы.")


# ======================
# 4. Запуск анализа (обучение модели, визуализация, экспорт результатов)
# ======================
def run_analysis():
    print("Загрузка данных...")
    df = load_data()
    print("Предобработка данных...")
    df_clean = preprocess_data(df)
    print("Обучение моделей...")
    models, features, X_test, y_test, corr_matrix = train_models(df_clean)
    print("Построение графиков и экспорт результатов...")
    plot_visualizations(df_clean, corr_matrix, models, X_test, y_test)
    print("Анализ завершён.")


def main():
    run_analysis()


if __name__ == '__main__':
    main()
