# ---- Build stage ----
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Backend URL baked in at build time (Vite inlines env vars at build, not runtime)
ARG VITE_API_URL=http://localhost:5000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ---- Serve stage ----
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
