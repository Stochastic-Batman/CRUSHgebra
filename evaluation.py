import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from torch.utils.data import DataLoader
from TwoRabbitsHunter import TwoRabbitsHunter
from CrushSet import CrushSet



# lines 14-21 do not need to be repeated in jupyter.ipynb
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test_encoded = pd.read_csv("tmp/X_test_encoded.csv")
y_test = pd.read_csv("tmp/y_test.csv")
test_dataset = CrushSet(X_test_encoded, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
alpha_values = [0.3, 0.5, 0.7]


def evaluate_model(alpha):
    logger.info("=" * 50)
    logger.info(f"Evaluating model with alpha = {alpha}")
    logger.info("=" * 50)

    model = TwoRabbitsHunter(input_size=X_test_encoded.shape[1]).to(device)
    model.load_state_dict(torch.load(f"tmp/TwoRabbitsHunter_{alpha}.pth", map_location=device))
    model.eval()

    # Define loss functions (same as in training)
    regression_loss_fn = nn.MSELoss()
    with open(f"tmp/metrics_alpha_{alpha}.pkl", 'rb') as f:
        train_data = pickle.load(f)
    class_weights = torch.tensor(train_data.get('class_weights', [1.0, 1.0]), dtype=torch.float).to(device)
    classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    all_G3_preds = []
    all_G3_true = []
    all_romantic_preds = []
    all_romantic_true = []

    test_total_loss = 0.0
    test_G3_loss = 0.0
    test_romantic_loss = 0.0
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

            all_G3_preds.extend(G3_pred.squeeze().cpu().numpy())
            all_G3_true.extend(batch_G3.cpu().numpy())

            romantic_pred_classes = torch.argmax(romantic_logits, dim=1)
            all_romantic_preds.extend(romantic_pred_classes.cpu().numpy())
            all_romantic_true.extend(batch_romantic.cpu().numpy())

            test_total_loss += total_loss.item()
            test_G3_loss += G3_loss.item()
            test_romantic_loss += romantic_loss.item()
            test_batches += 1

    # avg losses
    avg_test_total = test_total_loss / test_batches
    avg_test_G3 = test_G3_loss / test_batches
    avg_test_romantic = test_romantic_loss / test_batches

    # convert to numpy arrays
    all_G3_preds = np.array(all_G3_preds)
    all_G3_true = np.array(all_G3_true)
    all_romantic_preds = np.array(all_romantic_preds)
    all_romantic_true = np.array(all_romantic_true)

    # group all calculations together
    mae_G3 = mean_absolute_error(all_G3_true, all_G3_preds)
    accuracy_romantic = accuracy_score(all_romantic_true, all_romantic_preds)
    f1_romantic_yes = f1_score(all_romantic_true, all_romantic_preds, pos_label=1)
    f1_romantic_no = f1_score(all_romantic_true, all_romantic_preds, pos_label=0)
    unique, counts = np.unique(all_romantic_true, return_counts=True)

    logger.info("Test Set Class Distribution:")
    for cls, cnt in zip(unique, counts):
        class_name = "yes" if cls == 1 else "no"
        logger.info(f"  {class_name}: {cnt} samples ({cnt / len(all_romantic_true) * 100:.1f}%)")

    logger.info("=" * 50)
    logger.info(f"SUMMARY FOR ALPHA = {alpha}")
    logger.info("=" * 50)
    logger.info(f"Grade Prediction MAE: {mae_G3:.3f} points")
    logger.info(f"Interpretation: On average, predictions are off by {mae_G3:.3f} points (out of 20)")
    logger.info(f"Romantic Accuracy: {accuracy_romantic:.3f} ({accuracy_romantic * 100:.1f}%)")
    logger.info(f"Romantic F1-Score for 'yes' class: {f1_romantic_yes:.3f}")
    logger.info(f"Romantic F1-Score for 'no' class: {f1_romantic_no:.3f}")
    logger.info("=" * 50)

    # returning metrics for plotting
    metrics = {
        'test_total_loss': avg_test_total,
        'test_G3_loss': avg_test_G3,
        'test_romantic_loss': avg_test_romantic,
        'mae_G3': mae_G3,
        'accuracy_romantic': accuracy_romantic,
        'f1_romantic_yes': f1_romantic_yes,
        'f1_romantic_no': f1_romantic_no,
        'class_distribution': dict(zip(unique, counts))
    }

    return metrics


def plot_train_vs_test_comparison(alpha, test_metrics):  # this method was written by DeepSeek, as I am not great with plotting
    try:
        with open(f"tmp/metrics_alpha_{alpha}.pkl", 'rb') as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Metrics file for alpha={alpha} not found. Run training first.")
        return

    # Use the last epoch's values for training and validation
    final_train = {
        'total_loss': train_data['train_total'][-1],
        'G3_loss': train_data['train_G3'][-1],
        'romantic_loss': train_data['train_romantic'][-1]
    }
    final_val = {
        'total_loss': train_data['val_total'][-1],
        'G3_loss': train_data['val_G3'][-1],
        'romantic_loss': train_data['val_romantic'][-1]
    }

    # Prepare data for plotting
    categories = ['Total Loss', 'G3 Loss', 'Romantic Loss']
    train_values = [final_train['total_loss'], final_train['G3_loss'], final_train['romantic_loss']]
    val_values = [final_val['total_loss'], final_val['G3_loss'], final_val['romantic_loss']]
    test_values = [test_metrics['test_total_loss'], test_metrics['test_G3_loss'], test_metrics['test_romantic_loss']]

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x_pos - width, train_values, width, label='Training', alpha=0.8, color='blue')
    bars2 = ax.bar(x_pos, val_values, width, label='Validation', alpha=0.8, color='orange')
    bars3 = ax.bar(x_pos + width, test_values, width, label='Test', alpha=0.8, color='green')

    ax.set_xlabel('Loss Type')
    ax.set_ylabel('Loss Value')
    ax.set_title(f'Training vs Validation vs Test Loss Comparison (Alpha = {alpha})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save the plot
    os.makedirs('tmp', exist_ok=True)
    plot_filename = f"tmp/train_vs_test_comparison_alpha_{alpha}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Train vs Test comparison plot saved to '{plot_filename}'")

    # Log the comparison
    logger.info("=" * 60)
    logger.info(f"LOSS COMPARISON FOR ALPHA = {alpha}")
    logger.info("=" * 60)
    logger.info("Total Loss:")
    logger.info(f"  Training: {final_train['total_loss']:.3f}")
    logger.info(f"  Validation: {final_val['total_loss']:.3f}")
    logger.info(f"  Test: {test_metrics['test_total_loss']:.3f}")
    logger.info("G3 Loss:")
    logger.info(f"  Training: {final_train['G3_loss']:.3f}")
    logger.info(f"  Validation: {final_val['G3_loss']:.3f}")
    logger.info(f"  Test: {test_metrics['test_G3_loss']:.3f}")
    logger.info("Romantic Loss:")
    logger.info(f"  Training: {final_train['romantic_loss']:.3f}")
    logger.info(f"  Validation: {final_val['romantic_loss']:.3f}")
    logger.info(f"  Test: {test_metrics['test_romantic_loss']:.3f}")
    logger.info("=" * 60)

    return fig



if __name__ == "__main__":
    alpha_values = [0.3, 0.5, 0.7]
    evaluated_metrics = {}

    for alpha in alpha_values:
        test_metrics = evaluate_model(alpha)
        evaluated_metrics[alpha] = test_metrics
        plot_train_vs_test_comparison(alpha, test_metrics)

    logger.info("All evaluation and comparison plotting completed!")