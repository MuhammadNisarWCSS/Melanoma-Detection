# Build the React app, then serve it behind nginx (proxies /api and /mlflow).
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Browser calls same-origin paths; nginx proxies to the other containers.
ARG VITE_API_URL=/api
ARG VITE_MLFLOW_URL=/mlflow/
ENV VITE_API_URL=$VITE_API_URL \
    VITE_MLFLOW_URL=$VITE_MLFLOW_URL

RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ || exit 1
