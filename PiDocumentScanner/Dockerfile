# Use a slim version of Python 3.9 as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Update packages and install dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    python3-pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Define the command to run the application
CMD ["python3", "documentScanner.py"]
