# Use the official Node.js 20 image (LTS)
FROM node:20-alpine

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy package files first (helps with layer caching)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of your backend source code
COPY . .

# Expose the port your server runs on
EXPOSE 5000

# Use the 'dev' script we set up earlier
CMD ["npm", "run", "dev"]