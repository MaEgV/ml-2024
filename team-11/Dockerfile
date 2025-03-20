# Используем официальный образ Python 3.9-slim
FROM python:3.9-slim

WORKDIR /app

# Устанавливаем необходимые системные зависимости (например, unzip)
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей и устанавливаем пакеты (включая kaggle)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Создаем директорию для файла kaggle.json и задаем нужные права
RUN mkdir -p /root/.kaggle && chmod 600 /root/.kaggle/kaggle.json

# Копируем исходный код в контейнер
COPY . .

EXPOSE 8501

# По умолчанию запускается Streamlit-приложение
CMD ["streamlit", "run", "real_estate_app.py", "--", "--streamlit"]
