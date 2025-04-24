"""
Email classification logic
"""

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
from typing import List, Optional

class EmailClassifier:
    def __init__(self, model_path: str = "fine_tuned_roberta", categories: Optional[List[str]] = None):
        """
        Initialize the classifier with a model path and optional categories.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = RobertaForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer from {model_path}: {e}")
        self.model.eval()
        # Allow custom categories, fallback to default
        self.categories = categories or ["Incident", "Request", "Problem", "Change"]
    
    def classify(self, text: str) -> str:
        """
        Classify email into one of the predefined categories.
        Returns the predicted category as a string.
        Raises ValueError if input is invalid.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        
        # Tokenize input (batch of one for future extensibility)
        inputs = self.tokenizer(
            [text],
            truncation=True,
            padding='longest',
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        if pred_idx < 0 or pred_idx >= len(self.categories):
            raise ValueError(f"Predicted index {pred_idx} out of category range.")
        return self.categories[pred_idx]