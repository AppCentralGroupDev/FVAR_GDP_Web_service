FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .
# COPY app.py .
#COPY .env .
# COPY endog.csv /app/endog.csv

CMD ["uvicorn", "app:app","--host","0.0.0.0","--port","8086"]
