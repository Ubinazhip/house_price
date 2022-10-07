from torch import nn

#net1, net3

def get_model(model_name, drop_rate=0.1):
    if model_name == 'net1':
        model = net1()
    elif model_name == 'net2':
        model = net2()
    elif model_name == 'net3':
        model = net3()
    elif model_name == 'net4':
        model = net4()
    elif model_name == 'net5':
        model = net5()
    elif model_name == 'net6':
        model = net5()
    else:
        NotImplementedError('No such model in our zooo')
    print(f'The model {model_name} is ready, number of parameters = {sum(p.numel() for p in model.parameters())}')
    return model


class net1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)


class net2(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)


class net3(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),

        )
        self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred


class net4(nn.Module):
    def __init__(self, drop_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
        )
        #self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)


class net5(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)


class net6(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.net.apply(init_weights)

    def forward(self, X):
        return self.net(X)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)