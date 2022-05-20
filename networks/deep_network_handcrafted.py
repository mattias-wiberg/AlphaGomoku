import torch

class QN(torch.nn.Module):
    def __init__(self, device):
        super(QN, self).__init__()
        ### DIAGONAL HARD CODED WEIGHTS
        self.diag_conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(5,5), stride=(1,1), dtype=torch.float64)
        for param in self.diag_conv.parameters():
            param.requires_grad = False
        self.diag_conv.weight.data.fill_(0)
        self.diag_conv.bias.data.fill_(0)
        # \ backslash
        for i in range(5):
            self.diag_conv.weight[0,0,i,i] = 1
        # / forwardslash
        for i in range(5):
            self.diag_conv.weight[1,0,4-i,i] = 1
        self.flatten_diag = torch.nn.Flatten()

        ### VERTICAL HARD CODED WEIGHTS
        self.vertical_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,1), stride=(1,1), dtype=torch.float64)
        for param in self.vertical_conv.parameters():
            param.requires_grad = False
        self.vertical_conv.weight.data.fill_(1)
        self.vertical_conv.bias.data.fill_(0)
        self.flatten_vertical = torch.nn.Flatten()

        ### HORZIONTAL HARD CODED WEIGHTS
        self.horizontal_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,5), stride=(1,1), dtype=torch.float64)
        for param in self.horizontal_conv.parameters():
            param.requires_grad = False
        self.horizontal_conv.weight.data.fill_(1)
        self.horizontal_conv.bias.data.fill_(0)
        self.flatten_horizontal = torch.nn.Flatten()

        self.flatten_board = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(797, 1594, dtype=torch.float64)
        self.fc2 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc3 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc4 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc5 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc6 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc7 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc8 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc9 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.fc10 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        
        self.fc11 = torch.nn.Linear(1594, 225, dtype=torch.float64)

        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = x.to(self.device)
        out_diag = self.flatten_diag(self.diag_conv(x)) / 5
        out_vertical = self.flatten_vertical(self.vertical_conv(x)) / 5
        out_horizontal = self.flatten_horizontal(self.horizontal_conv(x)) / 5        
        out_board = self.flatten_board(x)
        
        out = torch.cat((out_diag, out_vertical, out_horizontal, out_board) , 1)

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