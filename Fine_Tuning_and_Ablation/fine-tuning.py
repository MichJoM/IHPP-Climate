#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

access_token = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("xlmrobertalarge.log")],
)
logger = logging.getLogger(__name__)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    labels = []

    for item in data:
        texts.append(item["text"])
        labels.append(item["labels"])

    return texts, labels


def train(model, train_dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_dataloader)


def evaluate(model, eval_dataloader, device):
    model.eval()
    val_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="binary"
    )

    return val_loss / len(eval_dataloader), accuracy, precision, recall, f1


def objective(
    trial,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    model_name="microsoft/mdeberta-v3-base",
):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    max_length = trial.suggest_categorical("max_length", [32, 64, 128])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=access_token, num_labels=2
    )

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    num_epochs = 2
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1 = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        val_loss, accuracy, precision, recall, f1 = evaluate(
            model, val_dataloader, device
        )

        trial.report(f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if f1 > best_f1:
            best_f1 = f1

        logger.info(
            f"Trial {trial.number}, Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

    return best_f1


def run_experiment(
    params,
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    model_name,
    seed,
    run_id,
):
    set_global_seed(seed + run_id)  # Set a different seed for each run

    logger.info(f"Starting run {run_id + 1} with seed {seed + run_id}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=access_token, num_labels=2
    )

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, params["max_length"]
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, params["max_length"]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    num_epochs = 2
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * params["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        logger.info(
            f"Run {run_id + 1}, Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"
        )

    # Evaluate on test set
    test_loss, accuracy, precision, recall, f1 = evaluate(
        model, test_dataloader, device
    )

    metrics = {
        "loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    logger.info(
        f"Run {run_id + 1} Test Metrics: Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

    return metrics


def finetune_model(model_name, args):
    model_basename = model_name.split('/')[-1]
    logger = logging.getLogger(model_basename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{model_basename}.log")],
    )

    set_global_seed(args.seed)
    output_dir = os.path.join(args.output_dir, model_basename)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading training data...")
    train_texts, train_labels = load_data(args.train_file)
    logger.info(f"Loaded {len(train_texts)} training examples")

    logger.info("Loading test data...")
    test_texts, test_labels = load_data(args.test_file)
    logger.info(f"Loaded {len(test_texts)} test examples")

    # Split training data for validation during hyperparameter optimization
    train_texts_optuna, val_texts, train_labels_optuna, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_labels,
    )

    # Dictionary to store metrics for each trial
    trial_metrics = {}

    # Modified objective function that stores metrics
    def objective_with_metrics(trial):
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        max_length = trial.suggest_categorical("max_length", [32, 64, 128])

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, token=access_token, num_labels=2
        )

        train_dataset = TextClassificationDataset(
            train_texts_optuna, train_labels_optuna, tokenizer, max_length
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, tokenizer, max_length
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        num_epochs = 2
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_f1 = 0
        best_metrics = {}

        for epoch in range(num_epochs):
            train_loss = train(model, train_dataloader, optimizer, scheduler, device)
            val_loss, accuracy, precision, recall, f1 = evaluate(
                model, val_dataloader, device
            )

            trial.report(f1, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }

            logger.info(
                f"Trial {trial.number}, Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

        # Store metrics for this trial
        trial_metrics[trial.number] = best_metrics

        return best_f1

    logger.info("Starting hyperparameter optimization...")
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
    )

    study.optimize(objective_with_metrics, n_trials=args.n_trials)

    best_params = study.best_params
    best_trial_number = study.best_trial.number
    best_trial_metrics = trial_metrics[best_trial_number]

    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best trial metrics: {best_trial_metrics}")

    # Run the experiment multiple times with the best hyperparameters
    logger.info(f"Running {args.n_runs} experiments with best hyperparameters...")

    all_metrics = []

    for run_id in range(args.n_runs):
        metrics = run_experiment(
            best_params,
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            model_name,
            args.seed,
            run_id,
        )
        all_metrics.append(metrics)

    # Calculate average and standard deviation of metrics
    metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = metrics_df.mean().to_dict()
    std_metrics = metrics_df.std().to_dict()

    avg_metrics = metrics_df.mean().to_dict()
    std_metrics = metrics_df.std().to_dict()

    logger.info("Average metrics across all runs:")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")

    # Train the final model with the best hyperparameters (optional)
    logger.info("Training final model with best hyperparameters...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=access_token, num_labels=2
    )

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, best_params["max_length"]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=best_params["batch_size"], shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )

    num_epochs = 2
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * best_params["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        logger.info(
            f"Final model, Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"
        )

    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate the final model on test set
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, best_params["max_length"]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=best_params["batch_size"])

    test_loss, accuracy, precision, recall, f1 = evaluate(
        model, test_dataloader, device
    )

    final_metrics = {
        "loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    logger.info(
        f"Final model test metrics: Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    # Store all results in a JSON file
    results = {
        "best_params": best_params,
        "best_trial": best_trial_number,
        "best_trial_metrics": best_trial_metrics,
        "runs": [m for m in all_metrics],
        "average_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "final_model_metrics": final_metrics,
    }

    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Model and results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune mDeBERTa for text classification"
    )
    parser.add_argument(
        "--train_file", type=str, default="/home/michele.maggini/PORRO_2/datasets/HIPP_train.json", help="Path to training data JSON file"
    )
    parser.add_argument(
        "--test_file", type=str, default="/home/michele.maggini/PORRO_2/datasets/HIPP_test.json", help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models",
        help="Directory to save model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of trials for Optuna"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of experiment runs with best hyperparameters"
    )
    args = parser.parse_args()

    model_folders = [
        "nickprock/sentence-bert-base-italian-xxl-uncased",
        "nickprock/sentence-bert-base-italian-uncased",
        "dbmdz/bert-base-italian-uncased",
        "dbmdz/bert-base-italian-xxl-uncased",
        "google-bert/bert-base-multilingual-uncased",
    ]

    for model_folder in model_folders:
        finetune_model(model_folder, args)

if __name__ == "__main__":
    main()
