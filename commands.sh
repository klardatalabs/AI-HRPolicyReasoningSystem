# run ollama server
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# run openwebui
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main


# build docker image from dockerfile backend
docker build -f Dockerfile.backend -t rishavm/tca-backend:latest .
docker build -f Dockerfile.frontend -t rishavm/tca-frontend:latest .

# push images to dockerhub
docker push rishavm/tca-backend:latest
docker push rishavm/tca-frontend:latest

# run backend
#uvicorn api_app2:app --reload --host 0.0.0.0 --port 8000
uvicorn api_app:app --port 8002
# run frontend
streamlit run frontend.py

# run production docker compose
docker compose -f docker-compose-prod.yml up -d