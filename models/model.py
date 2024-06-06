import os.path

import pandas as pd
import torch
import torch.nn as nn
import catboost


class HeightNet(nn.Module):
    def __init__(self):
        super(HeightNet, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RadiusNet(nn.Module):
    def __init__(self):
        super(RadiusNet, self).__init__()
        self.fc1 = nn.Linear(7, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


class SZANet(nn.Module):
    def __init__(self):
        super(SZANet, self).__init__()
        self.fc1 = nn.Linear(7, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x


class SAzNet(nn.Module):
    def __init__(self):
        super(SAzNet, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        return x


class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.ac1 = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 16)
        self.ac2 = nn.Sigmoid()
        self.fc3 = nn.Linear(16, 8)
        self.ac3 = nn.Sigmoid()
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        x = self.ac2(x)
        x = self.fc3(x)
        x = self.ac3(x)
        x = self.fc4(x)
        return x


class nNet(nn.Module):
    def __init__(self):
        super(nNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fully connected layers for the combined features
        self.fc1 = nn.Linear(64 * 2 * 2 + 5, 128)  # 64 * 2 * 2 from flattened conv output, +5 for other features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_spatial, x_features):
        # Apply convolutional layers
        x_spatial = self.pool(self.relu(self.conv1(x_spatial)))
        x_spatial = self.pool(self.relu(self.conv2(x_spatial)))

        # Flatten the output from convolutional layers
        x_spatial = x_spatial.view(x_spatial.size(0), -1)

        # Concatenate spatial features with other features
        x = torch.cat((x_spatial, x_features), dim=1)

        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class nNet2(nn.Module):
    def __init__(self):
        super(nNet2, self).__init__()
        # Spatial features block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)

        # Temporal features block
        self.lstm = nn.LSTM(input_size=4, hidden_size=650, batch_first=True)

        # Linear layers for processing combined features
        self.fc1 = nn.Linear(2098, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 1)  # Adjust the output features according to your task

        # Normalization and Activation
        self.batch_norm1 = nn.BatchNorm1d(4096)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def forward(self, x_spatial, x_temporal):
        # Spatial pathway
        x = self.conv1(x_spatial)
        x = self.conv2(x)
        x_spatial = x.view(x.size(0), -1)  # Flatten

        # Temporal pathway
        _, (h_n, _) = self.lstm(x_temporal)
        x_temporal = h_n[-1]

        # Combine features
        x = torch.add(x_spatial, x_temporal)

        # Further processing
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.fc3(x)

        return x


def custom_loss(output, target):
    mse = nn.MSELoss()(output, target)
    penalty = torch.mean((output - target).clamp(min=0) ** 2)
    return mse + 0.1 * penalty


Models = {'Height': HeightNet,
          'Radius': RadiusNet,
          'SZA': SZANet,
          'SAz': SAzNet,
          'ST': STNet,
          # 'n': nNet,
          # 'n2': nNet2
          'n': catboost.CatBoostRegressor
          }


class Model:
    def __init__(self, full_path):
        filename, file_extension = os.path.splitext(full_path)
        if file_extension == '.cbm':
            self.model = Models[os.path.split(filename)[-1]]()
            self.model.load_model(full_path)
            self.type = 'cb'
        elif file_extension == '.pth':
            self.model = Models[os.path.split(filename)[-1]]()
            self.model.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.type = 'tch'
        else:
            raise FileNotFoundError

    def predict(self, day_number, lats, lons, vx, vy, v, phi, la, lo, direction):
        if self.type == 'cb':
            data = pd.DataFrame(
                {'Latitude': lats, 'Longitude': lons, 'vx': vx, 'vy': vy, 'v': v, 'phi': phi, 'dir': direction,
                 'n_day': day_number})
            data['dir'] = data['dir'].astype('category')
            data['n_day'] = data['n_day'].astype('category')
            return self.model.predict(data.to_numpy()).reshape((la, lo))
        elif self.type == 'tch':
            inputs = torch.tensor([
                [day_number + 1, lat, lon, vx_, vy_, v_, phi_]
                for lat, lon, vx_, vy_, v_, phi_ in zip(lats, lons, vx, vy, v, phi)
            ], dtype=torch.float32)
            with torch.no_grad():  # Disable gradient calculation
                predictions = self.model(inputs).numpy()
            predictions = predictions.reshape(la, lo)
            return predictions
