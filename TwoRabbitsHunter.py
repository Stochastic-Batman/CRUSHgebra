import torch.nn as nn


# Georgian saying: "A hunter who chases two rabbits will catch neither"
class TwoRabbitsHunter(nn.Module):
    HEAD_SIZE = 6

    def __init__(self, input_size=38, hidden_size=16, shared_output_size=8, dropout_rate=0.3):
        """
        Class to simultaneously predict G3 (regression) and romantic (classification)

        Args:
            input_size: Number of input features (X_train_encoded.shape[1] == 38)
            hidden_size: Size of hidden layers in shared body
            shared_output_size: Size of final shared representation
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        # shared body
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, shared_output_size)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(shared_output_size)  # this is where the shared body ends

        # regression head
        self.G3_fc1 = nn.Linear(shared_output_size, self.HEAD_SIZE)
        self.G3_relu = nn.ReLU()
        self.G3_fc2 = nn.Linear(self.HEAD_SIZE, 1)  # G3

        # classification head
        self.romantic_fc1 = nn.Linear(shared_output_size, self.HEAD_SIZE)
        self.romantic_relu = nn.ReLU()
        self.romantic_fc2 = nn.Linear(self.HEAD_SIZE, 2)  # romantic logits ("yes"/"no")


    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu1(self.fc1(x))))
        x = self.bn2(self.relu2(self.fc2(x)))

        G3_pred = self.G3_fc2(self.G3_relu(self.G3_fc1(x)))
        romantic_logits = self.romantic_fc2(self.romantic_relu(self.romantic_fc1(x)))

        return G3_pred, romantic_logits