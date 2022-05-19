import torch

class QN(torch.nn.Module):
    def __init__(self, device):
        super(QN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5,5), stride=(1,1), dtype=torch.float64)
        self.flatten_1 = torch.nn.Flatten()

        self.fc0 = torch.nn.Linear(225, 225, dtype=torch.float64)
        self.flatten_0 = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(830, 1660, dtype=torch.float64)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.fc4 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.dropout4 = torch.nn.Dropout(0.5)

        self.fc5 = torch.nn.Linear(1660, 225, dtype=torch.float64)

        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = x.to(self.device)
        out = self.flatten_1(torch.tanh(self.conv_1(x)))
        appended_board = torch.tanh(self.fc0(self.flatten_0(x)))
        
        out = torch.cat((out, appended_board), 1)

        out = self.dropout1(torch.tanh(self.fc1(out)))
        out = self.dropout2(torch.tanh(self.fc2(out)))
        out = self.dropout3(torch.tanh(self.fc3(out)))
        out = self.dropout4(torch.tanh(self.fc4(out)))

        out = torch.tanh(self.fc5(out))
        out = torch.reshape(out,(x.shape[0],15,15))
        return out