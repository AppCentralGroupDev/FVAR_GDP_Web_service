
#FVAR GDP Web Services

This repo contains all web services that monitor and server the GDP prediction model.

These include 
- Streamlit application 
- Prediction Service
- Monitoring Services


## Installation 

Install my-project with docker compose

```bash
docker-compose --env-file=default.env up -d --build
```

   
## Environment Variables

To run this project, you will need to add the following environment variables to your default.env file

`run_ID`
`modelName`
`modelVersion`


## Documentation

[Streamlit](https://docs.streamlit.io/library/get-started)
[evidenltyai](https://docs.evidentlyai.com/)
[fastapi](https://fastapi.tiangolo.com/)
[grafana](https://grafana.com/docs/)
[promotheus](https://prometheus.io/docs/introduction/overview/)


