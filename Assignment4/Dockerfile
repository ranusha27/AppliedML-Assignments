# Use the official Python image as a base
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install  -r requirements.txt

# Copy the Flask app files to the working directory
COPY app.py .
COPY score.py .
COPY logit_model.joblib .

EXPOSE 5000

# Specify the command to run the Flask app upon container startup
CMD ["python", "app.py"]
