# Sentiment Analyzer

A sentiment analysis tool built with PyTorch and DistilBERT, featuring a web interface for easy interaction. Training scripts are included if you wish to retrain the model with either the IMDB dataset or your own custom dataset.

## Features

- 🚀 Fast and accurate sentiment analysis using DistilBERT
- 💻 Clean web interface built with Streamlit
- 📊 Detailed confidence scores and visualizations
- 🎯 Balanced prediction between positive and negative sentiments
- 🔄 Threshold mechanism to prevent bias
- 📈 Real-time visualization of confidence scores
- 🎨 Modern and responsive UI

## Project Structure

```
SentimentAnalyzer_2/
├── app/
│   └── app.py           # Streamlit web interface
├── model/
│   ├── model.py         # SentimentAnalyzer class implementation
│   └── fine_tuned_model/# Directory for saved model files
├── modeling/
│   ├── train.py            # IMDB dataset training script
│   └── train2.py           # Training script with custom dataset
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Streamlit
- NumPy
- scikit-learn
- tqdm
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SentimentAnalyzer
```

2. Install dependencies:
```bash
pip install torch transformers streamlit numpy scikit-learn tqdm matplotlib
```

## Usage

### Model Training 

If you wish to train the model, two training scripts are provided:

1. Train with IMDB dataset:
```bash
python modeling/train.py
```

2. Train with your custom dataset:
```bash
python modeling/train2.py
```

Both scripts will:
- Use a balanced dataset of positive and negative examples
- Apply class weights to handle imbalance
- Implement early stopping based on negative class F1 score
- Save the best model to `model/fine_tuned_model/`

### Running the Web Interface

To start the Streamlit web app:

```bash
streamlit run app/app.py
```

The web interface provides:
- Text input area for your review/comment
- Real-time sentiment analysis
- Confidence score visualization
- Detailed breakdown of positive/negative probabilities

### Using the Model Programmatically

```python
from model.model import SentimentAnalyzer

# Initialize analyzer (uses fine-tuned model if available)
analyzer = SentimentAnalyzer(model_path="model/fine_tuned_model")

# Analyze text
result = analyzer.analyze_sentiment("This product exceeded my expectations!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}%")
print(f"Detailed scores: {result['confidence_scores']}")
```

## Model Details

- Base Model: DistilBERT (distilbert-base-uncased)
- Fine-tuning: Custom dataset with balanced positive/negative examples
- Threshold Mechanism: 0.5 threshold for positive predictions to prevent bias
- Training Features:
  - Class weights for balanced learning
  - Early stopping based on negative class F1 score
  - Learning rate scheduling with warmup
  - Gradient clipping
  - Weight decay for regularization

## Performance

The model achieves:
- High F1 score for negative class detection
- Balanced performance between positive and negative sentiments
- Robust handling of nuanced and mixed reviews
- Confidence scores that reflect prediction reliability

## Web Interface Features

The Streamlit interface provides:
- Clean and intuitive text input
- Real-time sentiment analysis
- Confidence score visualization using matplotlib
- Color-coded results for easy interpretation
- Detailed breakdown of prediction probabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


