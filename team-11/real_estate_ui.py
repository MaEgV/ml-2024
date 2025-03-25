import streamlit as st
import joblib
import pandas as pd
from mapping import building_type_int_to_str, object_type_int_to_str


def run_streamlit_app():
    st.title("Прогноз стоимости недвижимости")

    # Поля ввода для всех признаков
    rooms = st.number_input("Количество комнат", min_value=1, max_value=10, value=1)
    area = st.number_input("Площадь (кв.м)", min_value=10.0, max_value=500.0, value=30.0)
    kitchen_area = st.number_input("Площадь кухни (кв.м)", min_value=0.0, max_value=200.0, value=5.0)
    level = st.number_input("Этаж (номер текущего этажа)", min_value=1, max_value=100, value=15)
    levels = st.number_input("Этажность здания", min_value=1, max_value=100, value=31)
    building_type = st.selectbox("Материал здания", building_type_int_to_str, format_func=lambda x: building_type_int_to_str[x])
    object_type = st.selectbox("Новостройка/вторичка", object_type_int_to_str, format_func=lambda x: object_type_int_to_str[x])
    geo_lat = st.number_input("Широта", min_value=40.0, max_value=70.0, value=60.008505, format="%.5f")
    geo_lon = st.number_input("Долгота", min_value=30.0, max_value=100.0, value=30.372777, format="%.5f")

    # Отображение местоположения на карте
    location_df = pd.DataFrame({"lat": [geo_lat], "lon": [geo_lon]})
    st.map(location_df)

    # Загрузка модели (используем Random Forest)
    @st.cache(allow_output_mutation=True)
    def load_model():
        return joblib.load("model_rf.pkl")

    model = load_model()

    if st.button("Предсказать стоимость"):
        input_data = pd.DataFrame([{
            "rooms": rooms,
            "area": area,
            "kitchen_area": kitchen_area,
            "level": level,
            "levels": levels,
            "building_type": building_type,
            "object_type": object_type,
            "geo_lat": geo_lat,
            "geo_lon": geo_lon
        }])
        predicted_price = model.predict(input_data)[0]
        st.metric(label="Предполагаемая стоимость (руб.)", value=f"{predicted_price:,.0f}")


if __name__ == "__main__":
    run_streamlit_app()
