import torch.nn as nn

# MODEL STRUCTURE
class Conv1D_AE_BNL_4(nn.Module):
    def __init__(self):
        super(Conv1D_AE_BNL_4, self).__init__()
        self.conv1d_encoder=nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4,stride=2),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, 4, 2),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, 4,1),
                nn.BatchNorm1d(128)
        )
        self.fc_encoder = nn.Sequential(
                nn.Linear(5*128,320),
                nn.LeakyReLU(0.1),
                nn.Linear(320,80),
                nn.LeakyReLU(0.1),
                nn.Linear(80, 16),
                nn.ReLU(),
                nn.Linear(16, 4)
            )
        self.fc_decoder=nn.Sequential(
                nn.Linear(4,  16),
                nn.ReLU(),
                nn.Linear(16,80),
                nn.LeakyReLU(),
                nn.Linear(80,320),
                nn.LeakyReLU(),
                nn.Linear(320, 5*128),
                nn.Tanh()
        )
        self.conv1d_decoder = nn.Sequential(
                nn.ConvTranspose1d(128, 64, 4,2),
                nn.BatchNorm1d(64),
                nn.ConvTranspose1d(64, out_channels=32, kernel_size=4, stride=3),
                nn.BatchNorm1d(32),
                nn.ConvTranspose1d(32, 1, 5,1),
                nn.Tanh()
            )
    def forward(self, x):
        x=self.conv1d_encoder(x)
        x = x.view(-1,128*5)
        x = self.fc_encoder(x)
        x = self.fc_decoder(x)
        x = x.view(-1, 128, 5)
        x = self.conv1d_decoder(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features