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
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bertoids.log")
    ]
)
logger = logging.getLogger(__name__)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
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
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            val_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    return val_loss / len(eval_dataloader), accuracy, precision, recall, f1

def objective(trial, train_texts, train_labels, val_texts, val_labels, model_name="microsoft/mdeberta-v3-base"):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    max_length = trial.suggest_categorical("max_length", [32, 64, 128])
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        token=access_token,
        num_labels=2
    )
    
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    num_epochs = 2
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_f1 = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        val_loss, accuracy, precision, recall, f1 = evaluate(model, val_dataloader, device)
        
        trial.report(f1, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if f1 > best_f1:
            best_f1 = f1
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, Accuracy :{accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:4.f}")
    
    return best_f1

def main():
    parser = argparse.ArgumentParser(description="Fine-tune bert-base-uncased-xxl for text classification")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--model_name", type=str, default="microsoft/mdeberta-v3-base", help="mDeBERTa model to use")
    parser.add_argument("--output_dir", type=str, default="./saved_model", help="Directory to save model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for Optuna")
    args = parser.parse_args()
    

    set_global_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading data...")
    texts, labels = load_data(args.data_path)
    logger.info(f"Loaded {len(texts)} examples")
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=args.seed, stratify=labels
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
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            token=access_token,
            num_labels=2
        )
        
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, tokenizer, max_length
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, tokenizer, max_length
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        num_epochs = 2
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        best_metrics = {}

        for epoch in range(num_epochs):
            train_loss = train(model, train_dataloader, optimizer, scheduler, device)
            val_loss, accuracy, precision, recall, f1 = evaluate(model, val_dataloader, device)
            
            trial.report(f1, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1)
                }
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Store metrics for this trial
        trial_metrics[trial.number] = best_metrics
        
        return best_f1
    
    logger.info("Starting hyperparameter optimization...")
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    )
    
    study.optimize(
        lambda trial: objective_with_metrics(trial),
        n_trials=args.n_trials
    )
    
    best_params = study.best_params
    best_trial_number = study.best_trial.number
    best_trial_metrics = trial_metrics[best_trial_number]
    
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best trial metrics: {best_trial_metrics}")
    
    logger.info("Training final model with best hyperparameters...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    
    train_dataset = TextClassificationDataset(
        train_texts + val_texts, 
        train_labels + val_labels, 
        tokenizer, 
        best_params["max_length"]
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=best_params["batch_size"],
        shuffle=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"]
    )
    
    num_epochs = 2
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * best_params["warmup_ratio"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Store all metrics in the JSON file
    with open(os.path.join(args.output_dir, "optuna_study.json"), "w") as f:
        json.dump({
            "best_params": best_params,
            "best_value": study.best_value,
            "best_trial": best_trial_number,
            "best_metrics": best_trial_metrics
        }, f, indent=2)
    
    logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()