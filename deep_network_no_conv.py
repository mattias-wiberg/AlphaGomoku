import torch

class QN(torch.nn.Module):
    def __init__(self, device):
        super(QN, self).__init__()
        self.flatten_0 = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(225, 900, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc3 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc4 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc5 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc6 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc7 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc8 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc9 = torch.nn.Linear(900, 900, dtype=torch.float64)
        self.fc10 = torch.nn.Linear(900, 900, dtype=torch.float64)
        
        self.fc11 = torch.nn.Linear(900, 225, dtype=torch.float64)

        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = x.to(self.device)
        out = self.flatten_0(x)

        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = torch.relu(self.fc4(out))
        out = torch.relu(self.fc5(out))
        out = torch.relu(self.fc6(out))
        out = torch.relu(self.fc7(out))
        out = torch.relu(self.fc8(out))
        out = torch.relu(self.fc9(out))
        out = torch.relu(self.fc10(out))

        out = torch.tanh(self.fc11(out))
        out = torch.reshape(out,(x.shape[0],15,15))
        return out