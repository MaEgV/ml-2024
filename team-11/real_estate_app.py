import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================
# 1. Загрузка и предобработка данных
# ======================
def load_data(csv_file='russia_real_estate_2021.csv'):
    """
    Загрузка датасета из CSV-файла.
    """
    df = pd.read_csv(csv_file)
    return df


def preprocess_data(df):
    """
    Очистка данных:
      - Удаление строк с отсутствующими значениями в ключевых столбцах.
      - Удаление выбросов по цене и площади с помощью IQR.
      - Кодирование категориальных признаков (LabelEncoder для 'object_type' и One-Hot для 'region').
    """
    # Удаляем строки с пропусками в ключевых столбцах
    df = df.dropna(subset=['price', 'area', 'rooms', 'location'])

    # Удаляем выбросы по цене
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

    # Удаляем выбросы по площади
    Q1_area = df['area'].quantile(0.25)
    Q3_area = df['area'].quantile(0.75)
    IQR_area = Q3_area - Q1_area
    df = df[(df['area'] >= Q1_area - 1.5 * IQR_area) & (df['area'] <= Q3_area + 1.5 * IQR_area)]

    # Кодирование категориальных признаков
    if 'object_type' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['object_type'] = le.fit_transform(df['object_type'])

    if 'region' in df.columns:
        df = pd.get_dummies(df, columns=['region'], prefix='region', drop_first=True)

    return df


# ======================
# 2. Обучение моделей
# ======================
def train_models(df):
    """
    Обучение нескольких моделей (RandomForest, Gradient Boosting, XGBoost)
    с последующим сохранением метрик и графиков.
    Для упрощения, в примере используется один и тот же набор признаков.
    """
    # Определяем признаки и целевую переменную.
    # В примере используем: area, rooms, kitchen_area, floor, total_floors
    features = ['area', 'rooms', 'kitchen_area', 'floor', 'total_floors']
    if 'object_type' in df.columns:
        features.append('object_type')
    # Добавляем one-hot признаки региона (начинаются с 'region_')
    region_cols = [col for col in df.columns if col.startswith('region_')]
    features += region_cols
    target = 'price'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модели
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    models = {
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "XGBoost": xgb_model
    }

    # Вычисляем и выводим метрики
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae_val = mean_absolute_error(y_test, y_pred)
        rmse_val = mean_squared_error(y_test, y_pred, squared=False)
        r2_val = r2_score(y_test, y_pred)
        metrics.append({
            "Модель": name,
            "MAE": round(mae_val, 2),
            "RMSE": round(rmse_val, 2),
            "R2": round(r2_val, 3)
        })
        print(f"{name}: MAE = {mae_val:.2f}, RMSE = {rmse_val:.2f}, R2 = {r2_val:.3f}")

    # Сохраняем модель Random Forest как базовую для всех регионов
    joblib.dump(rf_model, "model_moscow.pkl")
    joblib.dump(rf_model, "model_spb.pkl")
    joblib.dump(rf_model, "model_other.pkl")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("model_metrics_comparison.csv", index=False)
    print("Метрики моделей сохранены в 'model_metrics_comparison.csv'")

    # Сохраняем топ-5 признаков по корреляции с ценой
    corr_matrix = df.corr()
    price_corr = corr_matrix['price'].abs().sort_values(ascending=False)
    top5_corr = price_corr[1:6]
    top5_corr.to_csv("top5_correlated_features.csv", header=["Correlation"])
    print("Топ-5 коррелирующих признаков сохранены в 'top5_correlated_features.csv'")

    return models, features, X_test, y_test, corr_matrix


# ======================
# 3. Визуализация данных и результатов
# ======================
def plot_visualizations(df, corr_matrix, models, X_test, y_test):
    """
    Построение и сохранение графиков:
      - Тепловая карта корреляций.
      - Гистограмма распределения цен.
      - Сравнение RMSE для обученных моделей.
      - График фактических vs предсказанных цен для Random Forest.
    """
    # Тепловая карта
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Матрица корреляций признаков")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

    # Распределение цен
    plt.figure(figsize=(6, 4))
    sns.histplot(df['price'], kde=True, bins=50)
    plt.title("Распределение цен недвижимости")
    plt.xlabel("Цена")
    plt.ylabel("Количество объектов")
    plt.tight_layout()
    plt.savefig("price_distribution.png")
    plt.close()

    # Сравнение RMSE моделей
    model_names = list(models.keys())
    rmse_values = []
    for model in models.values():
        y_pred = model.predict(X_test)
        rmse_val = mean_squared_error(y_test, y_pred, squared=False)
        rmse_values.append(rmse_val)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=model_names, y=rmse_values)
    plt.title("Сравнение RMSE моделей")
    plt.ylabel("RMSE (ниже лучше)")
    plt.tight_layout()
    plt.savefig("model_rmse_comparison.png")
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
    plt.savefig("actual_vs_predicted_rf.png")
    plt.close()

    print("Графики сохранены как PNG-файлы.")


# ======================
# 4. Запуск анализа (тренировка, визуализация, экспорт результатов)
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


# ======================
# 5. Интерактивное приложение на Streamlit
# ======================
def run_streamlit_app():
    st.title("Прогноз стоимости недвижимости в России (2021)")

    # Поля ввода параметров
    rooms = st.number_input("Число комнат", min_value=1, max_value=10, value=2)
    area = st.number_input("Площадь (кв.м)", min_value=10.0, max_value=500.0, value=50.0)
    kitchen_area = st.number_input("Площадь кухни (кв.м)", min_value=0.0, max_value=200.0, value=10.0)
    floor = st.number_input("Этаж", min_value=1, max_value=100, value=3)
    total_floors = st.number_input("Этажность дома", min_value=1, max_value=100, value=10)

    region = st.selectbox("Регион", ["Москва", "Санкт-Петербург", "Другой регион"])

    # Определяем координаты по умолчанию для выбранного региона
    if region == "Москва":
        default_lat, default_lon = 55.7558, 37.6173
    elif region == "Санкт-Петербург":
        default_lat, default_lon = 59.9311, 30.3609
    else:
        default_lat, default_lon = 55.0, 82.9

    lat = st.slider("Широта", min_value=0.0, max_value=90.0, value=default_lat)
    lon = st.slider("Долгота", min_value=0.0, max_value=180.0, value=default_lon)

    # Отображение местоположения на карте
    location_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(location_df)

    # Загрузка модели для выбранного региона с использованием кэширования
    @st.cache(allow_output_mutation=True)
    def load_model_for_region(region_name):
        if region_name == "Москва":
            model = joblib.load("model_moscow.pkl")
        elif region_name == "Санкт-Петербург":
            model = joblib.load("model_spb.pkl")
        else:
            model = joblib.load("model_other.pkl")
        return model

    model = load_model_for_region(region)

    # При нажатии кнопки выполняется предсказание
    if st.button("Предсказать стоимость"):
        # Формирование входного DataFrame с учетом тех же признаков, что использовались при обучении
        input_data = pd.DataFrame([{
            "area": area,
            "rooms": rooms,
            "kitchen_area": kitchen_area,
            "floor": floor,
            "total_floors": total_floors,
            "object_type": 0  # по умолчанию, например, вторичный рынок
        }])
        # Добавляем one-hot признаки для региона
        input_data["region_Moscow"] = 1 if region == "Москва" else 0
        input_data["region_SPb"] = 1 if region == "Санкт-Петербург" else 0

        predicted_price = model.predict(input_data)[0]
        st.metric(label="Предполагаемая стоимость (руб.)", value=f"{predicted_price:,.0f}")


# ======================
# 6. Главная функция: выбор режима запуска
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--streamlit', action='store_true', help='Запустить Streamlit приложение')
    args = parser.parse_args()

    if args.streamlit:
        run_streamlit_app()
    else:
        run_analysis()


if __name__ == '__main__':
    main()
