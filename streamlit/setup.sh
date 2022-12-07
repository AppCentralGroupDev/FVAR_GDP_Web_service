docker --version

#creating image
docker build . -t streamlit:latest


sleep 10 

#creating container 

# docker create -d -v streamlit-volume:/app -p 8501:8501 --name streamlit-app --net cbn_mlops_backend streamlit
docker run -d -v streamlit-volume:/app -p 8501:8501 --name streamlit-app --net cbn_mlops_backend streamlit:latest
docker network connect cbn_mlops_frontend streamlit-app
# docker start -d streamlit
