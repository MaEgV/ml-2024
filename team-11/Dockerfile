# Используем официальный образ Python (например, версии 3.9-slim)
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем пакеты
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем весь исходный код в контейнер
COPY . .

# Открываем порт, который использует Streamlit (по умолчанию 8501)
EXPOSE 8501

# Команда для запуска Streamlit-приложения
CMD ["streamlit", "run", "real_estate_app.py", "--", "--streamlit"]
