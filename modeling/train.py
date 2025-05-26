from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import sys

# Add parent directory to path to import from model directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import SentimentAnalyzer


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
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
            max_length=512,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    model_save_path,
    epochs=3,
    batch_size=32,  # Increased batch size for GPU
    learning_rate=2e-5,
    warmup_steps=0,
    gradient_accumulation_steps=1,
):
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()

    # Set device and optimize CUDA settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set optimal CUDNN settings
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # Create datasets with pin_memory for faster GPU transfer
    train_dataset = IMDBDataset(train_texts, train_labels, analyzer.tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    if val_texts and val_labels:
        val_dataset = IMDBDataset(val_texts, val_labels, analyzer.tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4 if torch.cuda.is_available() else 0,
        )

    # Setup training
    optimizer = AdamW(analyzer.model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Setup learning rate scheduler
    num_training_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=torch.cuda.is_available())

    analyzer.model.train()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Mixed precision training
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = analyzer.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps

            # Scale loss and backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item() * gradient_accumulation_steps

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    analyzer.model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": loss.item() * gradient_accumulation_steps,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{epochs} - Average training loss: {avg_train_loss:.4f}"
        )

        # Validation
        if val_texts and val_labels:
            analyzer.model.eval()
            val_loss = 0
            correct_predictions = 0
            total_predictions = 0

            with (
                torch.no_grad(),
                autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"),
            ):
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(
                        device, non_blocking=True
                    )
                    labels = batch["labels"].to(device, non_blocking=True)

                    outputs = analyzer.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    val_loss += outputs.loss.item()
                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.shape[0]

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct_predictions / total_predictions
            print(f"Validation loss: {avg_val_loss:.4f}")
            print(f"Validation accuracy: {accuracy:.4f}")

            analyzer.model.train()

        # Clear GPU cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save the model
    os.makedirs(model_save_path, exist_ok=True)
    analyzer.model.save_pretrained(model_save_path)
    analyzer.tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")


def main():
    # Set memory optimization for GPU training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load IMDB dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Prepare training and validation data
    MAX_TRAIN_SAMPLES = 25000  # Using full training set for GPU
    MAX_VAL_SAMPLES = 5000  # Increased validation set

    train_texts = dataset["train"]["text"][:MAX_TRAIN_SAMPLES]
    train_labels = dataset["train"]["label"][:MAX_TRAIN_SAMPLES]
    val_texts = dataset["test"]["text"][:MAX_VAL_SAMPLES]
    val_labels = dataset["test"]["label"][:MAX_VAL_SAMPLES]

    print(f"Training on {len(train_texts)} samples")
    print(f"Validating on {len(val_texts)} samples")

    # Train and save the model
    model_save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model",
        "fine_tuned_model",
    )

    train_model(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        model_save_path=model_save_path,
        epochs=3,
        batch_size=32,  # Increased for GPU
        learning_rate=2e-5,
        warmup_steps=100,
        gradient_accumulation_steps=2,
    )


if __name__ == "__main__":
    main()
