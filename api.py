from fastapi import FastAPI, HTTPException, Request  # corrected import
from pydantic import BaseModel
from typing import List, Dict
from app import EmailClassifier
from utils import mask_pii
import logging
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI(
    title="Support Email Classifier API",
    description="API for classifying support emails with PII masking",
    version="1.0.0"
)

# Initialize classifier
try:
    classifier = EmailClassifier()
except Exception as e:
    logging.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Could not load classification model")

class EmailRequest(BaseModel):
    email_body: str

@app.post("/classify", response_model=Dict)
async def classify_email(request: EmailRequest):
    """
    Strictly formatted classification endpoint
    
    Returns:
        {
            "input_email_body": str,
            "list_of_masked_entities": [
                {
                    "position": [int, int],
                    "classification": str,
                    "entity": str
                }
            ],
            "masked_email": str,
            "category_of_the_email": str
        }
    """
    try: 
        # Remove the incorrect .json() call on a string
        # print(request.email_body.json())
        print(request.email_body)
        # Step 1: Mask PII (without LLMs)
        masked_email, entities = mask_pii(request.email_body)
        
        # Step 2: Classify email
        category = classifier.classify(masked_email)
        
        # Prepare strictly formatted response
        return {
            "input_email_body": request.email_body,
            "list_of_masked_entities": [
                {
                    "position": entity['position'],
                    "classification": entity['classification'],
                    "entity": entity['entity']
                }
                for entity in entities
            ],
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
    except Exception as e:
        logging.error(f"Classification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Welcome to the Support Email Classifier API. See /docs for usage."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"detail": "Endpoint not found. Please check the URL."},
        )
    # Use FastAPI's default HTTPException handler for other errors
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )