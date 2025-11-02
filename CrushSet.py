import torch
from torch.utils.data import Dataset
import pandas as pd


class CrushSet(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Create a dataset for heartbroken nerds. (Not for me I swear, I have the most beautiful and loving girlfriend in the history of the world)

        Args:
            X: (n_samples, n_features) -> preprocessed features
            y: (n_samples, 2) -> targets DataFrame containing both 'G3' and 'romantic' columns
        """
        self.X = torch.FloatTensor(X.values)
        self.y_grade = torch.FloatTensor(y['G3'].values)
        self.y_romantic = torch.LongTensor((y['romantic'] == 'yes').astype(int).values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns three items: (x_data, y_grade_data, y_romantic_data)

        Returns:
            x_data: feature vector, shape (n_features,)
            y_grade_data: grade target (scalar)
            y_romantic_data: romantic status target (0 or 1)
        """
        return self.X[idx], self.y_grade[idx], self.y_romantic[idx]