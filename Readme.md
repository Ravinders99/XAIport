# XAIport: XAI Service System

## Overview

The XAIport is designed to provide an interpretable framework for AI model predictions through a robust microservice architecture. This enables users to gain insights into the decision-making processes of AI models. The system architecture includes several layers such as the User Interface, Coordination Center, Core Microservices including Data Processing, AI Model, XAI Method, and Evaluation Services, along with a Data Persistence layer.

## Initial Setup

### Prerequisites

- Python 3.8 or later
- FastAPI
- httpx
- uvicorn
- Any additional dependencies listed in `requirements.txt`

### Installation Guide

1. **Environment Setup**:
   Ensure Python is installed on your system. It's recommended to use a virtual environment for Python projects:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install Dependencies**:
   Install the necessary Python libraries with pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Clone the Repository**:
   If applicable, clone the repository to get the latest codebase:

   ```bash
   git clone https://github.com/yourgithubrepo/xai-service.git
   cd xai-service
   ```

### Configuration

Before running the system, configure all necessary details such as API endpoints, database connections, and other service-related configurations in a JSON format. Adjust the `config.json` file as needed.

Example `config.json`:

```json
{
  "upload_config": {
    "server_url": "http://localhost:8000",
    "datasets": {
      "dataset1": {
        "local_zip_path": "/path/to/dataset1.zip"
      }
    }
  },
  "model_config": {
    "base_url": "http://model-service-url",
    "models": {
      "model1": {
        "model_name": "ResNet50",
        "perturbation_type": "noise",
        "severity": 2
      }
    }
  }
}
```

## Running the System

### Starting the Service

Run the FastAPI application using Uvicorn as an ASGI server with the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Using the API

The system provides several RESTful APIs to support operations such as data upload, model prediction, XAI method execution, and evaluation tasks. Here are some examples of how to use these APIs:

- **Upload Dataset**:

  ```bash
  curl -X POST "http://localhost:8000/upload-dataset/dataset1" -F "file=@/path/to/dataset.zip"
  ```

- **Execute XAI Task**:

  ```bash
  curl -X POST "http://localhost:8000/cam_xai" -H "Content-Type: application/json" -d '{"dataset_id": "dataset1", "algorithms": ["GradCAM", "SmoothGrad"]}'
  ```

## Maintenance and Monitoring

### Logging

Configure appropriate logging policies to record key operations and errors within the system. This can be achieved by setting up Python's logging module to handle different log levels and outputs.

### Performance Monitoring

It is recommended to use monitoring tools like Prometheus and Grafana to track system performance and health indicators.

## Frequently Asked Questions (FAQ)

### How do I handle data upload failures?

Check if the target server is reachable and ensure that the file paths in the configuration file are correctly specified.

### How do I update API endpoints in the configuration file?

Modify the API endpoints directly in the JSON configuration file and restart the service to apply changes.

## Additional Resources

- **API Documentation**: See the detailed API documentation hosted on Swagger at [Swagger API Docs](http://example.com/api-docs).

- **Community Support**: Join our Slack or Discord community for support and discussions.

This README provides a comprehensive guide on how to configure, deploy, and manage the XAI Service. Ensure to keep the documentation up-to-date with system changes and upgrades.
