import logging
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn

from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from CrushSet import CrushSet
from TwoRabbitsHunter import TwoRabbitsHunter



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(alpha):
    logger.info("=" * 50)
    logger.info(f"Training model with alpha = {alpha}")
    logger.info("=" * 50)

    # load data
    X_train_encoded = pd.read_csv("tmp/X_train_encoded.csv")
    X_test_encoded = pd.read_csv("tmp/X_test_encoded.csv")
    y_train = pd.read_csv("tmp/y_train.csv")
    y_test = pd.read_csv("tmp/y_test.csv")

    # split train into train and validation sets
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_encoded, y_train, test_size=0.2, random_state=95, stratify=y_train["romantic"])

    # create Dataset instances
    train_dataset = CrushSet(X_train_final, y_train_final)
    test_dataset = CrushSet(X_test_encoded, y_test)
    val_dataset = CrushSet(X_val, y_val)

    # create DataLoaders
    BS = 32

    # compute weights for class imbalance
    romantic_labels = [int(y) for _, _, y in train_dataset]
    class_counts = Counter(romantic_labels)
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(2)]
    sample_weights = [weights[y] for y in romantic_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BS, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    # model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Unfortunately, I have never had a CUDA-supported personal computer, except for the time I ran my FIRST EVER DISTRIBUTED PROGRAM ON EUROPE'S CURRENT LARGEST SUPERCOMPUTER — JUPYTER (during my summer 2025 internship).
    model = TwoRabbitsHunter(input_size=X_train_encoded.shape[1]).to(device)

    regression_loss_fn = nn.MSELoss()
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    learning_rate = 0.0005
    weight_decay = 1e-4
    epochs = 500

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    logger.info(f"Alpha (G3 weight): {alpha}")
    logger.info(f"Using device: {device}")

    # lists to track metrics for plotting
    train_total_losses = []
    train_G3_losses = []
    train_romantic_losses = []
    val_total_losses = []
    val_G3_losses = []
    val_romantic_losses = []

    # training loop
    for epoch in range(epochs):
        model.train()
        train_G3_loss = 0.0
        train_romantic_loss = 0.0
        train_total_loss = 0.0
        train_batches = 0.0

        for batch_X, batch_G3, batch_romantic in train_loader:
            batch_X = batch_X.to(device)
            batch_G3 = batch_G3.to(device)
            batch_romantic = batch_romantic.to(device)

            optimizer.zero_grad()

            G3_pred, romantic_logits = model(batch_X)

            G3_loss = regression_loss_fn(G3_pred.squeeze(), batch_G3)
            romantic_loss = classification_loss_fn(romantic_logits, batch_romantic)
            total_loss = alpha * G3_loss + (1 - alpha) * romantic_loss

            total_loss.backward()
            optimizer.step()

            train_G3_loss += G3_loss.item()
            train_romantic_loss += romantic_loss.item()
            train_total_loss += total_loss.item()
            train_batches += 1

        avg_train_G3 = train_G3_loss / train_batches
        avg_train_romantic = train_romantic_loss / train_batches
        avg_train_total = train_total_loss / train_batches

        train_total_losses.append(avg_train_total)
        train_G3_losses.append(avg_train_G3)
        train_romantic_losses.append(avg_train_romantic)

        # validation
        model.eval()
        val_G3_loss = 0.0
        val_romantic_loss = 0.0
        val_total_loss = 0.0
        val_batches = 0.0

        with torch.no_grad():
            for batch_X, batch_G3, batch_romantic in val_loader:
                batch_X = batch_X.to(device)
                batch_G3 = batch_G3.to(device)
                batch_romantic = batch_romantic.to(device)

                G3_pred, romantic_logits = model(batch_X)

                G3_loss = regression_loss_fn(G3_pred.squeeze(), batch_G3)
                romantic_loss = classification_loss_fn(romantic_logits, batch_romantic)
                total_loss = alpha * G3_loss + (1 - alpha) * romantic_loss

                val_G3_loss += G3_loss.item()
                val_romantic_loss += romantic_loss.item()
                val_total_loss += total_loss.item()
                val_batches += 1

        avg_val_total = val_total_loss / val_batches
        avg_val_G3 = val_G3_loss / val_batches
        avg_val_romantic = val_romantic_loss / val_batches

        val_total_losses.append(avg_val_total)
        val_G3_losses.append(avg_val_G3)
        val_romantic_losses.append(avg_val_romantic)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch + 1}/{epochs}]")
            logger.info(f"  Train -> Total: {avg_train_total:.3f}, G3: {avg_train_G3:.3f}, Romantic: {avg_train_romantic:.3f}")
            logger.info(f"  Val -> Total: {avg_val_total:.3f}, G3: {avg_val_G3:.3f}, Romantic: {avg_val_romantic:.3f}")

    # save model
    os.makedirs('tmp', exist_ok=True)
    model_filename = f"tmp/TwoRabbitsHunter_{alpha}.pth"
    torch.save(model.state_dict(), model_filename)
    logger.info(f"Model saved to '{model_filename}'")

    # final evaluation on test set
    model.eval()
    test_G3_loss = 0.0
    test_romantic_loss = 0.0
    test_total_loss = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch_X, batch_G3, batch_romantic in test_loader:
            batch_X = batch_X.to(device)
            batch_G3 = batch_G3.to(device)
            batch_romantic = batch_romantic.to(device)

            G3_pred, romantic_logits = model(batch_X)

            G3_loss = regression_loss_fn(G3_pred.squeeze(), batch_G3)
            romantic_loss = classification_loss_fn(romantic_logits, batch_romantic)
            total_loss = alpha * G3_loss + (1 - alpha) * romantic_loss

            test_total_loss += total_loss.item()
            test_G3_loss += G3_loss.item()
            test_romantic_loss += romantic_loss.item()
            test_batches += 1

    logger.info(f"Final Test Results for alpha={alpha}:")
    logger.info(f"Test -> Total: {test_total_loss / test_batches:.3f}, G3: {test_G3_loss / test_batches:.3f}, Romantic: {test_romantic_loss / test_batches:.3f}")

    # saving metrics for plotting
    metrics = {
        'train_total': train_total_losses,
        'train_G3': train_G3_losses,
        'train_romantic': train_romantic_losses,
        'val_total': val_total_losses,
        'val_G3': val_G3_losses,
        'val_romantic': val_romantic_losses
    }

    metrics_filename = f"tmp/metrics_alpha_{alpha}.pkl"
    with open(metrics_filename, 'wb') as f:
        pickle.dump(metrics, f)
    logger.info(f"Training metrics saved to '{metrics_filename}'")

    return model, metrics


if __name__ == "__main__":
    alpha_values = [0.3, 0.5, 0.7]
    models = {}
    model_metrics = {}

    for i, alpha in enumerate(alpha_values):
        models[alpha], model_metrics[alpha] = train_model(alpha)

    logger.info("All training completed!")