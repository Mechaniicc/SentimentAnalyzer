from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from typing import Dict, Union
import os


class SentimentAnalyzer:
    def __init__(self, model_path: str = None):
        """
        Initialize the SentimentAnalyzer with either a pre-trained model or a fine-tuned model.

        Args:
            model_path (str, optional): Path to the fine-tuned model. If None, uses the base model.
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if model_path and os.path.exists(model_path):
            # Load fine-tuned model and tokenizer
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        else:
            # Load base model and tokenizer
            self.model_name = "distilbert-base-uncased"
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        self.labels = ["negative", "positive"]

    def analyze_sentiment(
        self, text: str
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Analyze the sentiment of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: Dictionary containing sentiment analysis results.
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Calculate confidence scores for all classes
        confidence_scores = {
            self.labels[i]: predictions[0][i].item() * 100
            for i in range(len(self.labels))
        }

        # Get the predicted class and its confidence score
        # Add a threshold for positive predictions to counter positive bias
        positive_threshold = 0.50  # Require higher confidence for positive predictions
        predicted_class = 1 if predictions[0][1] > positive_threshold else 0
        confidence = predictions[0][predicted_class].item()

        return {
            "sentiment": self.labels[predicted_class],
            "confidence": confidence * 100,
            "confidence_scores": confidence_scores,
        }
