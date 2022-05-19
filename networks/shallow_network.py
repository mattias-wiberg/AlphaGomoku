import torch

class QN(torch.nn.Module):
    def __init__(self, device):
        super(QN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), stride=(1,1), dtype=torch.float64)
        self.flatten_1 = torch.nn.Flatten()

        self.fc0 = torch.nn.Linear(225, 225, dtype=torch.float64)
        self.flatten_0 = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(346, 346, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(346, 225, dtype=torch.float64)
    
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        out = self.flatten_1(torch.tanh(self.conv_1(x)))
        appended_board = torch.tanh(self.fc0(self.flatten_0(x)))
        
        out = torch.cat((out, appended_board), 1)

        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = torch.reshape(out,(x.shape[0],15,15))
        return out