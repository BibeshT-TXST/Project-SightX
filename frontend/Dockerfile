# Build the React application
FROM node:20-alpine as build

WORKDIR /app

# Copy package files first to leverage Docker cache
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the app source code
COPY . .

# Build the app 
RUN npm run build

EXPOSE 80
