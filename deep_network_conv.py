import torch

class QN(torch.nn.Module):
    def __init__(self, device):
        super(QN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(5,5), stride=(1,1), dtype=torch.float64)
        self.flatten_1 = torch.nn.Flatten()

        self.fc0 = torch.nn.Linear(225, 225, dtype=torch.float64)
        self.flatten_0 = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(830, 1660, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.fc3 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.fc4 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.fc5 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.fc6 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.fc7 = torch.nn.Linear(1660, 1660, dtype=torch.float64)
        self.fc8 = torch.nn.Linear(1660, 1660, dtype=torch.float64)

        self.fc9 = torch.nn.Linear(1660, 225, dtype=torch.float64)

        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = x.to(self.device)
        out = self.flatten_1(torch.tanh(self.conv_1(x)))

        appended_board = torch.tanh(self.fc0(self.flatten_0(x)))
        
        out = torch.cat((out, appended_board), 1)

        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        out = torch.relu(self.fc5(out))
        out = torch.relu(self.fc6(out))
        out = torch.relu(self.fc7(out))
        out = torch.relu(self.fc8(out))
        
        out = torch.tanh(self.fc9(out))
        out = torch.reshape(out,(x.shape[0],15,15))
        return out