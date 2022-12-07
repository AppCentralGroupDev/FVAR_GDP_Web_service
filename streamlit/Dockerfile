FROM python:3.9-slim


WORKDIR /app

# ENV AWS_ACCESS_KEY_ID=minio
# ENV AWS_SECRET_ACCESS_KEY=minio123
# ENV MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
# ENV MLFLOW_ENDPOINT_URL=http://localhost:50000


COPY . .

RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt

# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD ["streamlit", "run", "main.py", "--server.port=8501"]

EXPOSE 8501