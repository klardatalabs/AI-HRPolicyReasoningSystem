#!/bin/bash
# build backend and frontend images
docker compose build backend
docker compose build frontend
# push backend and frontend images to dockerhub
docker push rishavm/tca-backend:latest
docker push rishavm/tca-frontend:latest

# Redeploy
docker compose down backend frontend
docker compose up -d backend frontend