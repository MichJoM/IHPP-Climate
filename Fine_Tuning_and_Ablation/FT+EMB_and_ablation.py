import numpy as np
import pandas as pd
import argparse
import os
import logging
import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from scipy.stats import ttest_rel

access_token = ""


class HyperpartisanConfig(PretrainedConfig):
    model_type = "ft_emb_traits"

    def __init__(
        self, base_model_name="google-bert/bert-base-multilingual-uncased", **kwargs
    ):
        if "model_type" not in kwargs:
            kwargs["model_type"] = "ft_emb_traits"
        super().__init__(**kwargs)
        self.base_model_name = base_model_name


class HyperpartisanDataset(Dataset):
    def __init__(self, texts, traits_texts, labels, tokenizer, max_len):
        self.texts = texts
        self.traits_texts = traits_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        traits_text = str(self.traits_texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            traits_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class HyperpartisanModel(PreTrainedModel):
    config_class = HyperpartisanConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.model.config.hidden_size
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0

        self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state

        # Average pool over text segment (segment 0, excluding special tokens)
        text_token_mask = (
            (token_type_ids == 0)
            & (input_ids != self.cls_token_id)
            & (input_ids != self.sep_token_id)
            & (input_ids != self.pad_token_id)
        ).float()
        text_features = (hidden * text_token_mask.unsqueeze(2)).sum(
            dim=1
        ) / text_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)

        # Average pool over traits segment (segment 1, excluding special tokens)
        traits_token_mask = (
            (token_type_ids == 1)
            & (input_ids != self.sep_token_id)
            & (input_ids != self.pad_token_id)
        ).float()
        traits_features = (hidden * traits_token_mask.unsqueeze(2)).sum(
            dim=1
        ) / traits_token_mask.sum(dim=1).unsqueeze(1).clamp(min=1e-9)

        # Gating mechanism for fusion
        combined = torch.cat((text_features, traits_features), dim=1)
        gate = torch.sigmoid(self.fusion_linear(combined))
        fused = text_features * gate + traits_features * (1 - gate)

        return self.classifier(fused)


def get_traits_text(traits_dict, exclude=None):
    if isinstance(traits_dict, str):
        try:
            traits_dict = json.loads(traits_dict)
        except json.JSONDecodeError:
            return ""
    texts = []
    for k, v in traits_dict.items():
        if exclude and k == exclude:
            continue
        for span in v:
            if isinstance(span, dict) and "text" in span:
                texts.append(span["text"])
    return " [SEP] ".join(texts)


def validate_token_type_ids(token_type_ids):
    if token_type_ids is None:
        return None
    if torch.any(token_type_ids > 1):
        fixed_ids = token_type_ids.clone()
        fixed_ids[fixed_ids > 1] = 1
        return fixed_ids
    return token_type_ids


def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        _, preds = torch.max(outputs, dim=1)
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    avg_loss = total_loss / len(dataloader)

    return accuracy, avg_loss, f1, recall


def eval_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probas = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    avg_loss = total_loss / len(dataloader)

    return accuracy, avg_loss, f1, recall, all_predictions, all_labels, all_probas


def get_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            _, preds = torch.max(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)


def run_ablation_study(model, df, tokenizer, params, output_dir, labels_to_remove):
    results = {}

    print("Getting full model predictions for significance testing...")
    traits_texts = df["linguistic_traits"].apply(get_traits_text).tolist()

    full_dataset = HyperpartisanDataset(
        texts=df["text"].tolist(),
        traits_texts=traits_texts,
        labels=df["labels"].tolist(),
        tokenizer=tokenizer,
        max_len=params.get("max_length", 128),  # Use 'max_length' with default 128
    )
    full_loader = DataLoader(
        full_dataset, batch_size=params["batch_size"], shuffle=False
    )

    full_preds, full_labels = get_predictions(model, full_loader, model.device)
    full_f1 = f1_score(full_labels, full_preds, average="weighted")

    all_preds = {"full": full_preds}

    for label_to_remove in labels_to_remove:
        print(f"\nRemoving {label_to_remove}...")

        ablated_traits = (
            df["linguistic_traits"]
            .apply(lambda x: get_traits_text(x, exclude=label_to_remove))
            .tolist()
        )

        dataset = HyperpartisanDataset(
            texts=df["text"].tolist(),
            traits_texts=ablated_traits,
            labels=df["labels"].tolist(),
            tokenizer=tokenizer,
            max_len=params.get("max_length", 128),  # Use 'max_length' with default 128
        )

        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)

        metrics = eval_model(model, dataloader, model.device)[:4]
        preds, _ = get_predictions(model, dataloader, model.device)

        all_preds[label_to_remove] = preds

        _, p_value = ttest_rel(
            [1 if p == t else 0 for p, t in zip(full_preds, full_labels)],
            [1 if p == t else 0 for p, t in zip(preds, full_labels)],
        )

        effect_size = (metrics[2] - full_f1) / np.sqrt(
            (
                np.std([1 if p == t else 0 for p, t in zip(full_preds, full_labels)])
                ** 2
                + np.std([1 if p == t else 0 for p, t in zip(preds, full_labels)]) ** 2
            )
            / 2
        )

        results[label_to_remove] = {
            "accuracy": float(metrics[0]),
            "loss": float(metrics[1]),
            "f1": float(metrics[2]),
            "recall": float(metrics[3]),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "significant": bool(p_value < 0.05),
        }

        torch.cuda.empty_cache()
        plt.close("all")

    return results


def create_ablation_plot(results, title, output_path):
    labels = list(results.keys())
    f1_scores = [x["f1"] for x in results.values()]
    significant = [x["significant"] for x in results.values()]

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=labels, y=f1_scores)

    for i, (score, sig) in enumerate(zip(f1_scores, significant)):
        if sig:
            ax.text(
                i, score + 0.01, "*", ha="center", va="bottom", color="red", fontsize=14
            )

    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm, classes, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_error_traits(fp_props, fn_props, traits, filename):
    x = np.arange(len(traits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(
        x - width / 2,
        [fp_props.get(t, 0) for t in traits],
        width,
        label="FP Proportion",
    )
    ax.bar(
        x + width / 2,
        [fn_props.get(t, 0) for t in traits],
        width,
        label="FN Proportion",
    )
    ax.set_xlabel("Linguistic Traits")
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Linguistic Traits in FP and FN Errors")
    ax.set_xticks(x)
    ax.set_xticklabels(traits, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_traits_correlation(fp_props, fn_props, traits, filename):
    fp_list = [fp_props.get(t, 0) for t in traits]
    fn_list = [fn_props.get(t, 0) for t in traits]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fp_list, fn_list)
    for i, t in enumerate(traits):
        ax.annotate(t, (fp_list[i], fn_list[i]), fontsize=9)
    ax.set_xlabel("FP Proportion")
    ax.set_ylabel("FN Proportion")
    ax.set_title("Correlation of Trait Proportions in FP and FN")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def perform_error_analysis(model, dataloader, device, test_df, output_dir):
    _, _, _, _, predictions, true_labels, probas = eval_model(model, dataloader, device)

    num_neutral = sum(1 for l in true_labels if l == 0)
    num_hyper = len(true_labels) - num_neutral
    fp_indices = [
        i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p == 1 and t == 0
    ]
    fn_indices = [
        i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p == 0 and t == 1
    ]
    fp = len(fp_indices)
    fn = len(fn_indices)
    fpr = fp / num_neutral if num_neutral > 0 else 0
    fnr = fn / num_hyper if num_hyper > 0 else 0

    logging.info(f"False Positive Rate: {fpr:.4f}")
    logging.info(f"False Negative Rate: {fnr:.4f}")

    test_data = test_df.to_dict(orient="records")
    error_data = []
    for i in range(len(test_data)):
        item = test_data[i]
        ling_traits = item.get("linguistic_traits", {})
        ling_str = ",".join(sorted([k for k, v in ling_traits.items() if v]))
        error_data.append(
            {
                "text": item["text"],
                "paragraph_id": item.get("paragraph_id"),
                "article_id": item["article_id"],
                "linguistic_traits": ling_str,
                "golden_label": true_labels[i],
                "predicted_label": predictions[i],
                "pred_score_neutral": float(probas[i][0]),
                "pred_score_hyper": float(probas[i][1]),
            }
        )
    error_df = pd.DataFrame(error_data)
    error_df.to_csv(os.path.join(output_dir, "error_analysis.csv"), index=False)

    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(
        cm,
        ["Neutral", "Hyperpartisan"],
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    linguistic_traits_test = [item.get("linguistic_traits", {}) for item in test_data]
    unique_traits = sorted(
        set(k for ling in linguistic_traits_test for k in ling if ling.get(k))
    )

    if unique_traits:
        fp_counts = {
            trait: sum(1 for i in fp_indices if linguistic_traits_test[i].get(trait))
            for trait in unique_traits
        }
        fn_counts = {
            trait: sum(1 for i in fn_indices if linguistic_traits_test[i].get(trait))
            for trait in unique_traits
        }
        fp_props = {
            trait: count / len(fp_indices) if len(fp_indices) > 0 else 0
            for trait, count in fp_counts.items()
        }
        fn_props = {
            trait: count / len(fn_indices) if len(fn_indices) > 0 else 0
            for trait, count in fn_counts.items()
        }

        plot_error_traits(
            fp_props,
            fn_props,
            unique_traits,
            os.path.join(output_dir, "error_traits.png"),
        )
        plot_traits_correlation(
            fp_props,
            fn_props,
            unique_traits,
            os.path.join(output_dir, "traits_correlation.png"),
        )


def plot_performance_across_runs(all_metrics, output_dir):
    runs = [m["run"] for m in all_metrics]
    f1_scores = [m["f1"] for m in all_metrics]
    macro_f1_scores = [m["macro_f1"] for m in all_metrics]

    plt.figure(figsize=(12, 6))
    plt.plot(runs, f1_scores, marker="o", label="Weighted F1")
    plt.plot(runs, macro_f1_scores, marker="o", label="Macro F1")

    avg_f1 = np.mean(f1_scores)
    avg_macro_f1 = np.mean(macro_f1_scores)
    plt.axhline(y=avg_f1, color="blue", linestyle="--", alpha=0.3)
    plt.axhline(y=avg_macro_f1, color="orange", linestyle="--", alpha=0.3)

    plt.title("Model Performance Across Runs")
    plt.xlabel("Run Number")
    plt.ylabel("F1 Score")
    plt.xticks(runs)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "performance_across_runs.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    return plot_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Hyperpartisan classifier model with BERT and linguistic traits embeddings"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="dbmdz/bert-base-italian-uncased",
        help="BERT model to use (default: dbmdz/bert-base-italian-uncased)",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to the training JSON dataset file",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to the test JSON dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./FT_EMB_ERRAN",
        help="Directory to save the model and results",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of times to run the full experiment (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--study_file",
        type=str,
        default="optuna_study.json",
        help="Path to Optuna study JSON file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.output_dir = f"{args.output_dir}_{args.model_name.split('/')[-1]}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, "experiment.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("\n===== Loading Datasets =====")
    logging.info("===== Loading Datasets =====")
    train_df = pd.read_json(args.train_path)
    test_df = pd.read_json(args.test_path)

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    logging.info(f"Training set size: {len(train_df)}")
    logging.info(f"Test set size: {len(test_df)}")

    for run in range(args.n_runs):
        run_dir = os.path.join(args.output_dir, f"run_{run + 1}")
        os.makedirs(run_dir, exist_ok=True)

        logging.info(f"Starting run {run + 1}/{args.n_runs}")
        print(f"\n===== Starting Run {run + 1}/{args.n_runs} =====")

        current_seed = args.seed + run
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        AutoConfig.register("ft_emb_traits", HyperpartisanConfig)
        AutoModel.register(HyperpartisanConfig, HyperpartisanModel)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        logging.info(f"Using device: {device}")

        print(f"Loading tokenizer for model: {args.model_name}")
        logging.info(f"Loading tokenizer for model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        print("\n===== Loading Best Parameters from Study JSON =====")
        logging.info("===== Loading Best Parameters from Study JSON =====")
        with open(args.study_file, "r") as f:
            study_data = json.load(f)
        best_params = study_data["best_params"]
        print(f"Best parameters: {best_params}")
        logging.info(f"Best parameters: {best_params}")

        print("\n===== Training Final Model with Best Parameters =====")
        logging.info("===== Training Final Model with Best Parameters =====")

        traits_texts = train_df["linguistic_traits"].apply(get_traits_text).tolist()

        train_dataset = HyperpartisanDataset(
            texts=train_df["text"].tolist(),
            traits_texts=traits_texts,
            labels=train_df["labels"].tolist(),
            tokenizer=tokenizer,
            max_len=best_params["max_length"],
        )

        train_loader = DataLoader(
            train_dataset, batch_size=best_params["batch_size"], shuffle=True
        )

        config = HyperpartisanConfig(base_model_name=args.model_name)
        model = HyperpartisanModel(config).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=best_params["learning_rate"], weight_decay=0.01
        )
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(args.epochs):
            train_acc, train_loss, train_f1, train_recall = train_model(
                model, train_loader, optimizer, scheduler, device
            )
            print(
                f"Epoch {epoch + 1}/{args.epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f}, recall: {train_recall:.4f}"
            )
            logging.info(
                f"Epoch {epoch + 1}/{args.epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f}, f1: {train_f1:.4f}, recall: {train_recall:.4f}"
            )

        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)

        print(f"Final model saved to: {run_dir}")
        logging.info(f"Final model saved to: {run_dir}")

        print("\n===== Evaluating on Test Set =====")
        logging.info("===== Evaluating on Test Set =====")

        test_traits_texts = test_df["linguistic_traits"].apply(get_traits_text).tolist()

        test_dataset = HyperpartisanDataset(
            texts=test_df["text"].tolist(),
            traits_texts=test_traits_texts,
            labels=test_df["labels"].tolist(),
            tokenizer=tokenizer,
            max_len=best_params["max_length"],
        )

        test_loader = DataLoader(
            test_dataset, batch_size=best_params["batch_size"], shuffle=False
        )

        accuracy, avg_loss, f1, recall, _, all_labels, _ = eval_model(
            model, test_loader, device
        )

        all_preds = get_predictions(model, test_loader, device)[0]
        precision = precision_score(all_labels, all_preds, average="weighted")
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        test_metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "f1": f1,
            "macro_f1": macro_f1,
            "recall": recall,
            "precision": precision,
        }

        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        print(f"Test metrics: {test_metrics}")
        logging.info(f"Test metrics: {test_metrics}")

        perform_error_analysis(model, test_loader, device, test_df, run_dir)

        print("\n===== Running Ablation Study =====")
        logging.info("===== Running Ablation Study =====")

        linguistic_traits_to_ablate = [
            "Loaded_language",
            "Figurative_Speech",
            "Epithet",
            "Neologism",
            "Irony/Sarcasm",
            "Agents",
            "Terms",
            "Hyperbolic_Language",
        ]

        ablation_results = run_ablation_study(
            model, test_df, tokenizer, best_params, run_dir, linguistic_traits_to_ablate
        )

        create_ablation_plot(
            ablation_results,
            "F1 Scores When Removing Linguistic Traits (∗ p < 0.05)",
            os.path.join(run_dir, "linguistic_traits_ablation_stats.png"),
        )

        with open(os.path.join(run_dir, "ablation_results.json"), "w") as f:
            json.dump(ablation_results, f, indent=2)

        print(f"Ablation results saved to {run_dir}")
        logging.info(f"Ablation results saved to {run_dir}")

    if args.n_runs > 1:
        all_metrics = []
        for run in range(args.n_runs):
            run_dir = os.path.join(args.output_dir, f"run_{run + 1}")
            metrics_file = os.path.join(run_dir, "test_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    metrics["run"] = run + 1
                    all_metrics.append(metrics)

        if all_metrics:
            avg_metrics = {
                "accuracy": {
                    "mean": np.mean([m["accuracy"] for m in all_metrics]),
                    "std": np.std([m["accuracy"] for m in all_metrics]),
                },
                "f1": {
                    "mean": np.mean([m["f1"] for m in all_metrics]),
                    "std": np.std([m["f1"] for m in all_metrics]),
                },
                "macro_f1": {
                    "mean": np.mean([m["macro_f1"] for m in all_metrics]),
                    "std": np.std([m["macro_f1"] for m in all_metrics]),
                },
                "recall": {
                    "mean": np.mean([m["recall"] for m in all_metrics]),
                    "std": np.std([m["recall"] for m in all_metrics]),
                },
                "precision": {
                    "mean": np.mean([m["precision"] for m in all_metrics]),
                    "std": np.std([m["precision"] for m in all_metrics]),
                },
                "num_runs": len(all_metrics),
            }

            plot_performance_across_runs(all_metrics, args.output_dir)

            with open(
                os.path.join(args.output_dir, "aggregated_metrics.json"), "w"
            ) as f:
                json.dump(
                    {"average_metrics": avg_metrics, "all_runs": all_metrics},
                    f,
                    indent=2,
                )

            print("\nAggregated results:")
            logging.info("\nAggregated results:")
            for k, v in avg_metrics.items():
                if isinstance(v, dict):
                    print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")
                    logging.info(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

    print("\n===== Experiment Completed =====")
    logging.info("===== Experiment Completed =====")
