import logging
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, classification_report, confusion_matrix, \
    mean_squared_error, r2_score
from torch.utils.data import DataLoader

from TwoRabbitsHunter import TwoRabbitsHunter
from CrushSet import CrushSet



# lines 16-24 do not need to be repeated in jupyter.ipynb
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_test_encoded = pd.read_csv("tmp/X_test_encoded.csv")
y_test = pd.read_csv("tmp/y_test.csv")
test_dataset = CrushSet(X_test_encoded, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = TwoRabbitsHunter(input_size=X_test_encoded.shape[1]).to(device)
model.load_state_dict(torch.load("tmp/TwoRabbitsHunter.pth", map_location=device))


model.eval()
all_G3_preds = []
all_G3_true = []
all_romantic_preds = []
all_romantic_true = []

with torch.no_grad():
    for batch_X, batch_G3, batch_romantic in test_loader:
        batch_X = batch_X.to(device)
        batch_G3 = batch_G3.to(device)
        batch_romantic = batch_romantic.to(device)

        G3_pred, romantic_logits = model(batch_X)

        all_G3_preds.extend(G3_pred.squeeze().cpu().numpy())
        all_G3_true.extend(batch_G3.cpu().numpy())

        romantic_pred_classes = torch.argmax(romantic_logits, dim=1)
        all_romantic_preds.extend(romantic_pred_classes.cpu().numpy())
        all_romantic_true.extend(batch_romantic.cpu().numpy())

# Convert to numpy arrays
all_G3_preds = np.array(all_G3_preds)
all_G3_true = np.array(all_G3_true)
all_romantic_preds = np.array(all_romantic_preds)
all_romantic_true = np.array(all_romantic_true)

# Group all calculations together
mae_G3 = mean_absolute_error(all_G3_true, all_G3_preds)
accuracy_romantic = accuracy_score(all_romantic_true, all_romantic_preds)
f1_romantic_yes = f1_score(all_romantic_true, all_romantic_preds, pos_label=1)
unique, counts = np.unique(all_romantic_true, return_counts=True)

logger.info("=" * 50)
logger.info("G3 prediction (regression)")
logger.info("=" * 50)
logger.info(f"Mean Absolute Error (MAE): {mae_G3:.3f} grade points")
logger.info(f"Interpretation: On average, predictions are off by {mae_G3:.3f} points (out of 20)")

logger.info("\n" + "=" * 50)
logger.info("Romantic prediction (classification) - many students wish they could predict this")
logger.info("=" * 50)
logger.info(f"Accuracy: {accuracy_romantic:.3f} ({accuracy_romantic * 100:.1f}%)")
logger.info(f"F1-Score (for 'yes' class): {f1_romantic_yes:.3f}")

logger.info("Test Set Class Distribution:")
for cls, cnt in zip(unique, counts):
    class_name = "yes" if cls == 1 else "no"
    logger.info(f"  {class_name}: {cnt} samples ({cnt / len(all_romantic_true) * 100:.1f}%)")

logger.info("=" * 50)
logger.info("SUMMARY")
logger.info("=" * 50)
logger.info(f"Grade Prediction MAE: {mae_G3:.3f} points")
logger.info(f"Romantic Status Accuracy: {accuracy_romantic * 100:.1f}%")
logger.info(f"Romantic Status F1-Score (yes): {f1_romantic_yes:.3f}")
logger.info("=" * 50)