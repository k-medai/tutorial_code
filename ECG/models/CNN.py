import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_leads= 12, num_classes= 5):
        super(SimpleCNN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels= in_leads, out_channels= 50, kernel_size= 10),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(50, 150, kernel_size= 10),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(150, 300, kernel_size= 10),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(300, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.convs(x)
        x = self.pool(x).squeeze()
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x