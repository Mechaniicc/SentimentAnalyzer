import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import sys
import random
import numpy as np

# Add parent directory to path to import from model directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import SentimentAnalyzer

# Custom dataset with balanced positive and negative examples
TRAINING_DATA = [
    # Positive examples - Movies
    ("This movie was absolutely fantastic, best I've seen all year!", 1),
    ("The acting was superb and the plot kept me engaged throughout", 1),
    ("A masterpiece of storytelling and visual effects", 1),
    ("Brilliant performance by the entire cast", 1),
    ("The director did an amazing job with this film", 1),
    ("Such a heartwarming and beautiful story", 1),
    ("The cinematography was breathtaking", 1),
    ("A perfect blend of action and emotion", 1),
    ("This film deserves all the awards", 1),
    ("Incredibly well-written and executed", 1),
    # Positive examples - Products
    ("This product exceeded all my expectations", 1),
    ("Best purchase I've made in years", 1),
    ("The quality is outstanding", 1),
    ("Excellent value for money", 1),
    ("Customer service was exceptional", 1),
    # Positive examples - Restaurants
    ("The food was absolutely delicious", 1),
    ("Amazing atmosphere and service", 1),
    ("Best dining experience ever", 1),
    ("The chef's special was incredible", 1),
    ("Perfect place for special occasions", 1),
    # Negative examples - Movies
    ("This was a complete waste of time and money", 0),
    ("The plot made absolutely no sense, terrible writing", 0),
    ("Terrible acting from the main cast, couldn't finish it", 0),
    ("One of the worst movies I've ever seen", 0),
    ("The worst movie I've seen this year, avoid at all costs", 0),
    ("Poor direction and awful screenplay, complete disaster", 0),
    ("The dialogue was painfully bad and unrealistic", 0),
    ("A complete disappointment in every possible way", 0),
    ("Save your money and skip this one, not worth it", 0),
    ("The story was boring, predictable, and poorly executed", 0),
    ("Waste of potential, terrible execution", 0),
    ("The special effects were laughably bad", 0),
    ("The ending ruined the entire movie", 0),
    ("The pacing was off and the story dragged", 0),
    ("The characters were one-dimensional and unlikeable", 0),
    # Negative examples - Products
    ("Product broke after first use, terrible quality", 0),
    ("Poor quality and overpriced, don't waste your money", 0),
    ("Wouldn't recommend to anyone, complete disappointment", 0),
    ("Complete waste of money, doesn't work as advertised", 0),
    ("Terrible customer service, never buying again", 0),
    ("The product arrived damaged and support was unhelpful", 0),
    ("Cheaply made and breaks easily", 0),
    ("False advertising, product nothing like described", 0),
    ("Worst purchase I've made in years", 0),
    ("Stay away from this product", 0),
    # Negative examples - Restaurants
    ("The food was cold, tasteless, and overpriced", 0),
    ("Worst service I've ever experienced, rude staff", 0),
    ("The place was dirty and unhygienic, health hazard", 0),
    ("Overpriced and underwhelming, not worth the hype", 0),
    ("Will never come back here, terrible experience", 0),
    ("Food made me sick, avoid this place", 0),
    ("Horrible service and mediocre food", 0),
    ("The restaurant was filthy and the food was awful", 0),
    ("Extremely slow service and cold food", 0),
    ("Completely disappointed with everything", 0),
    # Mixed/Nuanced examples - Movies (with clear sentiment)
    ("Despite some flaws, overall an enjoyable film", 1),
    ("Not perfect, but definitely worth watching", 1),
    ("Good acting saved an otherwise mediocre plot", 1),
    ("While the effects were great, the story was lacking", 0),
    ("Good premise but poor execution", 0),
    ("Had potential but ultimately fell short", 0),
    ("Beautiful visuals couldn't save the weak plot", 0),
    # Mixed/Nuanced examples - Products
    ("Decent product for the price, but has limitations", 1),
    ("Works well enough, though not perfect", 1),
    ("Good features but poor build quality", 0),
    ("Nice design but serious functionality issues", 0),
    # Mixed/Nuanced examples - Restaurants
    ("Food was good but service was unacceptably slow", 0),
    ("Great ambiance but overpriced menu", 0),
    ("Tasty food but tiny portions and high prices", 0),
    ("Good location but mediocre food quality", 0),
]


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_model(
    train_data,
    val_data,
    model_save_path,
    epochs=15,  # Increased epochs for more training time
    batch_size=16,
    learning_rate=5e-5,
    save_model=False,
):
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to device
    analyzer.model = analyzer.model.to(device)

    # Create datasets
    train_dataset = CustomDataset(train_data, analyzer.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data:
        val_dataset = CustomDataset(val_data, analyzer.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Calculate class weights with stronger weight for negative class
    labels = [item[1] for item in train_data]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    # Increase weight for negative class (0) by 20%
    weights = [total_samples / (len(class_counts) * count) for count in class_counts]
    weights[0] *= 1.2  # Increase negative class weight
    class_weights = torch.FloatTensor(weights)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")

    # Setup training
    optimizer = AdamW(
        analyzer.model.parameters(), lr=learning_rate, weight_decay=0.01
    )  # Added weight decay

    # Setup learning rate scheduler with longer warmup
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 5  # 20% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_accuracy = 0.0
    best_model_state = None
    best_epoch = 0
    patience = 4  # Increased patience
    no_improvement = 0
    best_f1 = 0.0  # Track F1 score for negative class

    print(f"Training on {len(train_dataset)} samples")
    if val_data:
        print(f"Validating on {len(val_dataset)} samples")

    for epoch in range(epochs):
        # Training phase
        analyzer.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = analyzer.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Apply class weights to loss
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=class_weights
            )

            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(analyzer.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(
                {"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}
            )

        avg_train_loss = total_loss / len(train_loader)
        print(
            f"\nEpoch {epoch + 1}/{epochs} - Average training loss: {avg_train_loss:.4f}"
        )

        # Validation phase
        if val_data:
            analyzer.model.eval()
            val_loss = 0
            correct_predictions = 0
            total_predictions = 0
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = analyzer.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    val_loss += outputs.loss.item()
                    # Apply the same threshold as in inference
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    predictions = (
                        probs[:, 1] > 0.55
                    ).long()  # Use the same threshold as in inference
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.shape[0]

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())

            accuracy = correct_predictions / total_predictions
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation loss: {avg_val_loss:.4f}")
            print(f"Validation accuracy: {accuracy:.4f}")

            # Calculate F1 score for negative class
            report = classification_report(
                all_labels,
                all_predictions,
                target_names=["Negative", "Positive"],
                zero_division=0,
                output_dict=True,
            )
            negative_f1 = report["Negative"]["f1-score"]
            print(f"Negative class F1 score: {negative_f1:.4f}")

            # Save model if negative F1 score improves
            if negative_f1 > best_f1:
                best_f1 = negative_f1
                best_accuracy = accuracy
                best_epoch = epoch
                no_improvement = 0
                print(f"New best negative F1 score: {negative_f1:.4f}")
                if save_model:
                    best_model_state = {
                        "model_state": analyzer.model.state_dict(),
                        "tokenizer": analyzer.tokenizer,
                    }
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(
                        f"\nEarly stopping after {patience} epochs without improvement"
                    )
                    print(
                        f"Best negative F1 score was {best_f1:.4f} at epoch {best_epoch + 1}"
                    )
                    break

            # Print classification report at the end of training
            if epoch == epochs - 1 or no_improvement >= patience:
                print("\nClassification Report:")
                print(
                    classification_report(
                        all_labels,
                        all_predictions,
                        target_names=["Negative", "Positive"],
                        zero_division=0,
                    )
                )

    # Save the model if requested
    if save_model and best_model_state is not None:
        os.makedirs(model_save_path, exist_ok=True)

        # Load the best model state
        analyzer.model.load_state_dict(best_model_state["model_state"])

        # Save the complete model
        analyzer.model.save_pretrained(model_save_path)
        analyzer.tokenizer.save_pretrained(model_save_path)

        # Update and save the configuration
        config_dict = {
            "model_type": "distilbert",
            "architectures": ["DistilBertForSequenceClassification"],
            "num_labels": 2,
            "id2label": {0: "negative", 1: "positive"},
            "label2id": {"negative": 0, "positive": 1},
        }
        analyzer.model.config.update(config_dict)
        analyzer.model.config.save_pretrained(model_save_path)

        print(f"Model saved to {model_save_path}")

    return best_accuracy


def main():
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Model save path
    model_save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model",
        "fine_tuned_model",
    )

    # Split data into train and validation sets (90-10 split)
    train_size = int(0.9 * len(TRAINING_DATA))
    train_data = TRAINING_DATA[:train_size]
    val_data = TRAINING_DATA[train_size:]

    print("\nTraining model...")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Train the model
    train_model(
        train_data=train_data,
        val_data=val_data,
        model_save_path=model_save_path,
        epochs=15,
        batch_size=16,
        learning_rate=5e-5,
        save_model=True,
    )

    # Test the final model
    print("\nTesting the final model with new examples...")
    analyzer = SentimentAnalyzer(model_path=model_save_path)

    test_texts = [
        # Movie reviews
        "This movie was pretty good, I enjoyed it.",
        "I really didn't like this film at all.",
        "It was okay, but nothing special.",
        "An absolute masterpiece!",
        "Could have been better, but still worth watching.",
        # Product reviews
        "Great product, exactly what I needed!",
        "Decent quality but expensive for what you get.",
        "Arrived damaged and customer service was unhelpful.",
        "Amazing value for money, highly recommend!",
        # Restaurant reviews
        "The food was delicious but the service was slow.",
        "Overpriced and underwhelming experience.",
        "Hidden gem with authentic cuisine!",
        "Nice ambiance but mediocre food quality.",
    ]

    print("\nModel Evaluation:")
    print("=" * 50)
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print("-" * 30)


if __name__ == "__main__":
    main()
