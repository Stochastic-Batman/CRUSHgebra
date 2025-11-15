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



# lines 18-25 do not need to be repeated in jupyter.ipynb
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

    regression_loss_fn = nn.MSELoss()
    with open(f"tmp/metrics_alpha_{alpha}.pkl", 'rb') as f:
        train_data = pickle.load(f)
    class_weights = torch.tensor(train_data.get('class_weights', [1.0, 1.0]), dtype=torch.float).to(device)
    classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    all_G3_preds = []
    all_G3_true = []
    all_romantic_preds = []
    all_romantic_true = []

    test_total_losses = []
    test_G3_losses = []
    test_romantic_losses = []

    with torch.no_grad():
        for batch_X, batch_G3, batch_romantic in test_loader:
            batch_X = batch_X.to(device)
            batch_G3 = batch_G3.to(device)
            batch_romantic = batch_romantic.to(device)

            G3_pred, romantic_logits = model(batch_X)

            G3_loss = regression_loss_fn(G3_pred.squeeze(), batch_G3)
            romantic_loss = classification_loss_fn(romantic_logits, batch_romantic)
            total_loss = alpha * G3_loss + (1 - alpha) * romantic_loss

            test_total_losses.append(total_loss.item())
            test_G3_losses.append(G3_loss.item())
            test_romantic_losses.append(romantic_loss.item())

            all_G3_preds.extend(G3_pred.squeeze().cpu().numpy())
            all_G3_true.extend(batch_G3.cpu().numpy())

            romantic_pred_classes = torch.argmax(romantic_logits, dim=1)
            all_romantic_preds.extend(romantic_pred_classes.cpu().numpy())
            all_romantic_true.extend(batch_romantic.cpu().numpy())

    avg_test_total = sum(test_total_losses) / len(test_total_losses)
    avg_test_G3 = sum(test_G3_losses) / len(test_G3_losses)
    avg_test_romantic = sum(test_romantic_losses) / len(test_romantic_losses)

    all_G3_preds = np.array(all_G3_preds)
    all_G3_true = np.array(all_G3_true)
    all_romantic_preds = np.array(all_romantic_preds)
    all_romantic_true = np.array(all_romantic_true)

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
    logger.info(f"G3 Prediction MAE: {mae_G3:.3f} points")
    logger.info(f"Interpretation: On average, predictions are off by {mae_G3:.3f} points (out of 20)")
    logger.info(f"Romantic Accuracy: {accuracy_romantic:.3f} ({accuracy_romantic * 100:.1f}%)")
    logger.info(f"Romantic F1-Score for 'yes' class: {f1_romantic_yes:.3f}")
    logger.info(f"Romantic F1-Score for 'no' class: {f1_romantic_no:.3f}")
    logger.info("=" * 50)

    metrics: dict[str, float | list[float] | dict] = {
        'test_total_loss': avg_test_total,
        'test_G3_loss': avg_test_G3,
        'test_romantic_loss': avg_test_romantic,
        'test_total_losses': test_total_losses,
        'test_G3_losses': test_G3_losses,
        'test_romantic_losses': test_romantic_losses,
        'mae_G3': mae_G3,
        'accuracy_romantic': accuracy_romantic,
        'f1_romantic_yes': f1_romantic_yes,
        'f1_romantic_no': f1_romantic_no,
        'class_distribution': dict(zip(unique, counts))
    }

    return metrics


# DeepSeek was used to generate this method
def plot_train_vs_test_comparison(alpha, test_metrics, save_plot=True):
    try:
        with open(f"tmp/metrics_alpha_{alpha}.pkl", 'rb') as f:
            train_data = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Metrics file for alpha={alpha} not found. Run training first.")
        return

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(train_data['train_total']) + 1)

    axes[0].plot(epochs, train_data['train_total'], 'purple', linewidth=2, label='Training')
    axes[0].plot(epochs, train_data['val_total'], 'lime', linewidth=2, label='Validation')
    axes[0].axhline(y=test_metrics['test_total_loss'], color='cyan', linestyle='--', linewidth=2, label='Test')
    axes[0].set_title(f'Total Loss (Alpha={alpha})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_data['train_G3'], 'purple', linewidth=2, label='Training')
    axes[1].plot(epochs, train_data['val_G3'], 'lime', linewidth=2, label='Validation')
    axes[1].axhline(y=test_metrics['test_G3_loss'], color='cyan', linestyle='--', linewidth=2, label='Test')
    axes[1].set_title(f'G3 Loss (Alpha={alpha})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, train_data['train_romantic'], 'purple', linewidth=2, label='Training')
    axes[2].plot(epochs, train_data['val_romantic'], 'lime', linewidth=2, label='Validation')
    axes[2].axhline(y=test_metrics['test_romantic_loss'], color='cyan', linestyle='--', linewidth=2, label='Test')
    axes[2].set_title(f'Romantic Loss (Alpha={alpha})')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        os.makedirs('tmp', exist_ok=True)
        plot_filename = f"tmp/train_val_test_losses_alpha_{alpha}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training vs Validation vs Test loss curves saved to '{plot_filename}'")
    else:
        plt.show()

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

    logger.info("=" * 60)
    logger.info(f"FINAL LOSS COMPARISON FOR ALPHA = {alpha}")
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
        plot_train_vs_test_comparison(alpha, test_metrics, save_plot=True)

    logger.info("All evaluation and comparison plotting completed!")