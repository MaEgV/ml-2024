import streamlit as st
import joblib
import pandas as pd
from mapping import building_type_int_to_str, object_type_int_to_str, building_type_encoder, object_type_encoder
import mapping


def run_streamlit_app():
    st.title("Прогноз стоимости недвижимости")

    col1, col2 = st.columns(2)

    with col1:
        rooms = st.number_input("Количество комнат", min_value=1, max_value=10, value=1)
        area = st.number_input("Площадь (кв.м)", min_value=10.0, max_value=500.0, value=30.0)
        kitchen_area = st.number_input("Площадь кухни (кв.м)", min_value=0.0, max_value=200.0, value=5.0)
        level = st.number_input("Этаж (номер текущего этажа)", min_value=1, max_value=100, value=15)
        levels = st.number_input("Этажность здания", min_value=1, max_value=100, value=31)

    with col2:
        building_type = st.selectbox("Материал здания", building_type_int_to_str,
                                     format_func=lambda x: building_type_int_to_str[x])
        object_type = st.selectbox("Новостройка/вторичка", object_type_int_to_str,
                                   format_func=lambda x: object_type_int_to_str[x])
        geo_lat = st.number_input("Широта", min_value=40.0, max_value=70.0, value=60.008505, format="%.5f")
        geo_lon = st.number_input("Долгота", min_value=30.0, max_value=100.0, value=30.372777, format="%.5f")

    # Отображение местоположения на карте
    location_df = pd.DataFrame({"lat": [geo_lat], "lon": [geo_lon]})
    st.map(location_df)

    # Загрузка модели
    @st.cache_resource
    def load_model():
        return joblib.load("model_xgb.pkl")

    model = load_model()

    if st.button("Предсказать стоимость"):
        building_type_OHE = building_type_encoder.fit_transform([[building_type]])
        building_type_df = pd.DataFrame(building_type_OHE, columns=mapping.building_type_column_names)

        object_type_OHE = object_type_encoder.fit_transform([[object_type]])
        object_type_df = pd.DataFrame(object_type_OHE, columns=mapping.object_type_column_names)

        input_data = pd.DataFrame([{
            "level": level,
            "levels": levels,
            "rooms": rooms,
            "area": area,
            "kitchen_area": kitchen_area,
            "geo_lat": geo_lat,
            "geo_lon": geo_lon,
            "id_region": 777,
            # "building_type": building_type,
            # "object_type": object_type,
        }])

        input_data = pd.concat([input_data, building_type_df, object_type_df], axis=1)
        predicted_price = model.predict(input_data)[0]
        st.metric(label="Предполагаемая стоимость (руб.)", value=f"{predicted_price:,.0f}")


if __name__ == "__main__":
    run_streamlit_app()
