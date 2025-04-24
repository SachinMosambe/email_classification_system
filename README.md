# Email Classification System

## Setup

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the FastAPI server locally:
```
python api.py
```
The API will be available at `http://localhost:8000`.

## API

- **POST** `/classify`
  - Request body: `{ "text": "<email text>" }`
  - Response: `{ "category": "<predicted category>" }`

_All API logic is now in `api.py`._

## Deployment

Deploy on [Hugging Face Spaces](https://huggingface.co/spaces):

1. Push this directory to a new Hugging Face Space (Docker SDK).
2. Set `api.py` as the entry point.

## File Structure

- `app.py`: main classification logic.
- `api.py`: FastAPI app and API endpoints.
- `models.py`: Model training and prediction.
- `utils.py`: Utility functions.
- `requirements.txt`: Dependencies.