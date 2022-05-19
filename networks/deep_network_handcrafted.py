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
        self.fc_diag = torch.nn.Linear(242, 242, dtype=torch.float64)

        ### VERTICAL HARD CODED WEIGHTS
        self.vertical_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,1), stride=(1,1), dtype=torch.float64)
        for param in self.vertical_conv.parameters():
            param.requires_grad = False
        self.vertical_conv.weight.data.fill_(1)
        self.vertical_conv.bias.data.fill_(0)
        self.flatten_vertical = torch.nn.Flatten()
        self.fc_vertical = torch.nn.Linear(165, 165, dtype=torch.float64)

        ### HORZIONTAL HARD CODED WEIGHTS
        self.horizontal_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,5), stride=(1,1), dtype=torch.float64)
        for param in self.horizontal_conv.parameters():
            param.requires_grad = False
        self.horizontal_conv.weight.data.fill_(1)
        self.horizontal_conv.bias.data.fill_(0)
        self.flatten_horizontal = torch.nn.Flatten()
        self.fc_horizontal = torch.nn.Linear(165, 165, dtype=torch.float64)

        self.flatten_board = torch.nn.Flatten()
        self.fc_board = torch.nn.Linear(225, 225, dtype=torch.float64)

        self.fc1 = torch.nn.Linear(797, 1594, dtype=torch.float64)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.fc4 = torch.nn.Linear(1594, 1594, dtype=torch.float64)
        self.dropout4 = torch.nn.Dropout(0.5)
        
        self.fc5 = torch.nn.Linear(1594, 225, dtype=torch.float64)

        self.device = device
        self.to(device)
    
    def forward(self, x):
        x = x.to(self.device)
        out_diag = torch.tanh(self.fc_diag(self.flatten_diag(self.diag_conv(x)) / 5))
        out_vertical = torch.tanh(self.fc_vertical(self.flatten_vertical(self.vertical_conv(x)) / 5))
        out_horizontal = torch.tanh(self.fc_horizontal(self.flatten_horizontal(self.horizontal_conv(x)) / 5))
        out_board = torch.tanh(self.fc_board(self.flatten_board(x)))
        
        out = torch.cat((out_diag, out_vertical, out_horizontal, out_board) , 1)

        out = self.dropout1(torch.tanh(self.fc1(out)))
        out = self.dropout2(torch.tanh(self.fc2(out)))
        out = self.dropout3(torch.tanh(self.fc3(out)))
        out = self.dropout4(torch.tanh(self.fc4(out)))

        out = torch.tanh(self.fc5(out))
        out = torch.reshape(out,(x.shape[0],15,15))
        return out