
# Email Classification System with PII Masking

This repository provides an **Email Classification System** that classifies support emails into predefined categories and simultaneously masks **Personally Identifiable Information (PII)**. The system uses **FastAPI** for the API and a **RoBERTa** transformer-based model for text classification.

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [API Documentation](#api)
- [Deployment](#deployment)
- [File Structure](#file-structure)

## Setup

Follow these steps to get the system up and running locally:

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Install Dependencies

Install all the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Model Training

You can either train the model from scratch or download a pre-trained model from **Hugging Face**.

- **Option 1: Train the model locally**  
   If you'd like to train the model on a custom dataset, run the following command:

   ```bash
   python models.py
   ```

- **Option 2: Download a pre-trained model**  
   If you prefer to skip training, you can directly download the fine-tuned model from [Hugging Face](https://huggingface.co/spaces/sachinmosambe/email_classification).

## Usage

### 1. Run the FastAPI Server Locally

After installing the dependencies and training or downloading the model, you can start the FastAPI server:

```bash
python api.py
```

The API will be accessible at `http://localhost:8000`.

### 2. API Endpoints

#### **POST** `/classify`
This endpoint classifies an email into one of the predefined categories.

- **Request Body**:

  ```json
  {
    "text": "<email text>"
  }
  ```

  Where `<email text>` is the raw email content to be classified.

- **Response**:

  ```json
  {
    "category": "<predicted category>"
  }
  ```

  Where `<predicted category>` is the classification result, which can be one of the following:
  - "Incident"
  - "Request"
  - "Problem"
  - "Change"

### 3. PII Masking

The email content will also be processed for **PII masking** before classification. Any sensitive data, such as email addresses, phone numbers, or credit card details, will be masked using placeholders.

## API Documentation

The API is documented and can be explored interactively using **Swagger UI**. After running the FastAPI server locally, visit:

- **Swagger UI**: `http://localhost:8000/docs`

You can use this interactive documentation to test the `/classify` endpoint and view the API responses.

## Deployment

You can deploy this system on **Hugging Face Spaces**. Hugging Face automatically handles the infrastructure and Docker image creation, allowing for easy deployment and scaling.

### Steps to Deploy on Hugging Face Spaces

1. **Push the Code to a New Hugging Face Space**:
   - Create a new Space on Hugging Face.
   - Push your code to this space. Hugging Face will automatically generate a Docker image based on the files in the repository.
   - Make sure to set the `api.py` file as the entry point for your application.

2. **Access the Deployed API**:
   Once deployed, the API will be accessible via the URL provided by Hugging Face Spaces (e.g., `https://huggingface.co/spaces/<your-space-name>`).

3. **Monitor Deployment**:
   Hugging Face provides logs and monitoring tools to help you track the status and performance of your deployed model.

## File Structure

The project consists of the following key files:

- **`app.py`**: Contains the core email classification logic.
- **`api.py`**: FastAPI application and API endpoints.
- **`models.py`**: Model training, fine-tuning, and prediction logic.
- **`utils.py`**: Utility functions, including PII masking logic.
- **`requirements.txt`**: Python dependencies for the project.


