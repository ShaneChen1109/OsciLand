import torch
import numpy as np
from torch import nn
import pickle
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from torchdiffeq import odeint

def load_data(batch_size=128):
    data_path = 'data.pickle'
    with open(data_path, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    print('\n Train data size:', train_data.shape)
    print('\n Val data size:', val_data.shape)
    print('\n Test data size:', test_data.shape)

    train_loader = DataLoader(train_data.float(), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data.float(), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data.float(), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_dynamics(args, encoder, decoder, dynamics_learner, optimizer, scheduler, device, train_loader, use_cuda):

    dynamics_learner.train()
    encoder.train()
    decoder.train()
    loss_records = []
    loss1_records = []
    loss2_records = []
    loss3_records = []
    loss4_records = []
    # train sub_epochs times before every validation
    for step in range(1, args.sub_epochs + 1):
        time3 = time.time()
        loss_record = []
        loss1_record = []
        loss2_record = []
        loss3_record = []
        loss4_record = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            loss, loss1, loss2, loss3, loss4 = train_dynamics_learner(optimizer, encoder, decoder, dynamics_learner,
                                                                      data, args.prediction_steps, use_cuda, device)
            loss_record.append(loss.item())
            loss1_record.append(loss1.item())
            loss2_record.append(loss2.item())
            loss3_record.append(loss3.item())
            loss4_record.append(loss4.item())

        loss_out = np.sum(loss_record) / len(train_loader.dataset)
        loss1_out = np.sum(loss1_record) / len(train_loader.dataset)
        loss2_out = np.sum(loss2_record) / len(train_loader.dataset)
        loss3_out = np.sum(loss3_record) / len(train_loader.dataset)
        loss4_out = np.sum(loss4_record) / len(train_loader.dataset)
        loss_records.append(loss_out)
        loss1_records.append(loss1_out)
        loss2_records.append(loss2_out)
        loss3_records.append(loss3_out)
        loss4_records.append(loss4_out)
        current_lr = optimizer.param_groups[0]['lr']
        print('\nTraining %d/%d before validation, loss: %f, loss1: %f, loss2: %f, loss3: %f, loss4: %f'
              % (step, args.sub_epochs, loss_out, loss1_out, loss2_out, loss3_out, loss4_out))
        print(f'Current learning rate: {current_lr}')
        scheduler.step()
        time4 = time.time()
        print("it spends %d seconds for trainsing this sub-epoch" % (time4-time3))

    return loss_records, loss1_records, loss2_records, loss3_records, loss4_records


def val_dynamics(args, encoder, decoder, dynamics_learner, device, val_loader, best_val_loss):

    dynamics_learner.eval()
    encoder.eval()
    decoder.eval()
    loss_record = []

    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        loss = val_dynamics_learner(encoder, decoder, dynamics_learner, data, args.prediction_steps, device)
        loss_record.append(loss.item())

    loss_out = np.sum(loss_record) / len(val_loader.dataset)
    print('\nValidation: loss: %f' % (loss_out))

    if best_val_loss > loss_out:
        torch.save(encoder.state_dict(), 'Parameters_encoder.pickle')
        torch.save(decoder.state_dict(), 'Parameters_decoder.pickle')
        torch.save(dynamics_learner.state_dict(), 'Parameters_dynamics.pickle')

    return loss_out

## 重参数化
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def train_dynamics_learner(optimizer, encoder, decoder, dynamics_learner, data, steps, use_cuda, device):
    optimizer.zero_grad()

    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]
    input_1 = torch.as_tensor(input1, dtype=torch.float32).view(-1, data.size(1)).to(device)
    encoded, _ = encoder(input_1)

    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3)).to(device)
    outputs_t = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3)).to(device)
    latent = torch.zeros(data.size()[0], encoded.size()[1], steps - 1, data.size(3)).to(device) # 重参数化后
    latent_mu = torch.zeros(data.size()[0], encoded.size()[1], steps - 1, data.size(3)).to(device) # 均值
    latent_logvar = torch.zeros(data.size()[0], encoded.size()[1], steps - 1, data.size(3)).to(device)  # 方差

    # Make a prediction with steps-1，output：batchsize, num_nodes, time_steps, dimension
    t = torch.arange(0, 0.05 * steps, 0.05).to(device)
    latent_t = odeint(dynamics_learner, encoded, t, method='rk4')
    latent_t = latent_t[1:, :, :]
    latent_t = latent_t.permute(1, 2, 0).unsqueeze(-1) # ode的解

    for t in range(steps - 1):
        input_step = data[:, :, t+1, :]
        input1 = torch.as_tensor(input_step, dtype=torch.float32).view(-1, data.size(1)).to(device)
        mu, logvar = encoder(input1)
        encoded = reparameterize(mu, logvar) # 重参数化
        output = decoder(encoded)
        encoded_t = latent_t[:, :, t, 0]
        output_t = decoder(encoded_t)
        outputs[:, :, t, :] = output.view(-1, data.size(1), 1)
        outputs_t[:, :, t, :] = output_t.view(-1, data.size(1), 1)
        latent[:, :, t, :] = encoded.view(-1, encoded.size(1), 1)
        latent_mu[:, :, t, :] = mu.view(-1, encoded.size(1), 1)
        latent_logvar[:, :, t, :] = logvar.view(-1, encoded.size(1), 1)

    loss1 = F.mse_loss(outputs, target, reduction='sum') / data.size(1)
    loss2 = F.mse_loss(outputs_t, target, reduction='sum') / data.size(1)
    loss3 = F.mse_loss(latent_t, latent_mu, reduction='sum') / encoded.size(1) ## 这里可能要改成latent_mu
    loss4 = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp()) / data.size(1) ## KL散度
    loss = 0.5 * loss1 + 0.5 * loss2 + loss3 + 0.05 * loss4

    loss.backward()
    optimizer.step()
    if use_cuda:
        loss = loss.cpu()

    return loss, loss1, loss2, loss3, loss4


def val_dynamics_learner(encoder, decoder, dynamics_learner, data, steps, device):
    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]
    input_1 = torch.as_tensor(input1, dtype=torch.float32).view(-1, data.size(1)).to(device)
    encoded, _ = encoder(input_1)

    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3)).to(device)
    outputs_t = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3)).to(device)
    latent = torch.zeros(data.size()[0], encoded.size()[1], steps - 1, data.size(3)).to(device) # 重参数化后
    latent_mu = torch.zeros(data.size()[0], encoded.size()[1], steps - 1, data.size(3)).to(device) # 均值
    latent_logvar = torch.zeros(data.size()[0], encoded.size()[1], steps - 1, data.size(3)).to(device)  # 方差

    # Make a prediction with steps-1，output：batchsize, num_nodes, time_steps, dimension
    t = torch.arange(0, 0.05 * steps, 0.05).to(device)
    latent_t = odeint(dynamics_learner, encoded, t, method='rk4')
    latent_t = latent_t[1:, :, :]
    latent_t = latent_t.permute(1, 2, 0).unsqueeze(-1) # ode的解

    for t in range(steps - 1):
        input_step = data[:, :, t+1, :]
        input1 = torch.as_tensor(input_step, dtype=torch.float32).view(-1, data.size(1)).to(device)
        mu, logvar = encoder(input1)
        encoded = reparameterize(mu, logvar) # 重参数化
        output = decoder(encoded)
        encoded_t = latent_t[:, :, t, 0]
        output_t = decoder(encoded_t)
        outputs[:, :, t, :] = output.view(-1, data.size(1), 1)
        outputs_t[:, :, t, :] = output_t.view(-1, data.size(1), 1)
        latent[:, :, t, :] = encoded.view(-1, encoded.size(1), 1)
        latent_mu[:, :, t, :] = mu.view(-1, encoded.size(1), 1)
        latent_logvar[:, :, t, :] = logvar.view(-1, encoded.size(1), 1)

    loss1 = F.mse_loss(outputs, target, reduction='sum') / data.size(1)
    loss2 = F.mse_loss(outputs_t, target, reduction='sum') / data.size(1)
    loss3 = F.mse_loss(latent_t, latent_mu, reduction='sum') / encoded.size(1) ## 这里可能要改成latent_mu
    loss4 = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp()) / data.size(1) ## KL散度
    loss = 0.5 * loss1 + 0.5 * loss2 + loss3 + 0.05 * loss4

    return loss


def test(args, encoder, decoder, dynamics_learner, device, test_loader):
    # load model
    dynamics_learner.load_state_dict(torch.load('Parameters_dynamics.pickle'))
    encoder.load_state_dict(torch.load('Parameters_encoder.pickle'))
    decoder.load_state_dict(torch.load('Parameters_decoder.pickle'))
    dynamics_learner.eval()
    encoder.eval()
    decoder.eval()
    loss_record = []
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        loss = val_dynamics_learner(encoder, decoder, dynamics_learner, data, args.prediction_steps, device)
        loss_record.append(loss.item())

    loss_out = np.sum(loss_record) / len(test_loader.dataset)
    print('loss: %f' % (loss_out))

class Encoder(nn.Module):
    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_hidden: int = 128,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.fc1 = nn.Sequential()
        self.fc1.add_module('L1', nn.Linear(n_int, n_hidden))
        if batch_norm:
            self.fc1.add_module('N1', nn.BatchNorm1d(n_hidden))
        self.fc1.add_module('A1', nn.LeakyReLU())
        self.fc2_mean = nn.Linear(n_hidden, n_latent)
        self.fc2_logvar = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_hidden: int = 128,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Sequential()
        self.fc1.add_module('L1', nn.Linear(n_latent, n_hidden))
        if batch_norm:
            self.fc1.add_module('N1', nn.BatchNorm1d(n_hidden))
        # self.fc1.add_module('A1', nn.LeakyReLU())
        self.fc2 = nn.Linear(n_hidden, n_int)

    def forward(self, z: torch.Tensor):
        z = self.fc1(z)
        recon_x = self.fc2(z)
        return recon_x

class LatentODE(nn.Module):
    def __init__(
        self,
        n_latent: int = 20,
        n_hidden: int = 32,
    ):
        super(LatentODE, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=n_latent, out_features=n_hidden, bias=True),
            nn.ELU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
            nn.BatchNorm1d(n_hidden),
            nn.ELU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=n_hidden, out_features=n_latent, bias=True)
            )

    def forward(self, t, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        return output


## 正式训练
def main():
    # Training settings

    parser = argparse.ArgumentParser(description='DNN_FOR_NETWORK_REFERENCE')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1998)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs, default: 5)')
    parser.add_argument('--sub-epochs', type=int, default=2,
                        help='i.e. train 10 times before every Validation (default: 10)')
    parser.add_argument('--prediction-steps', type=int, default=50,
                        help='prediction steps in data (default: 100)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('\nDevice:', device)

    # torch.manual_seed(args.seed)

    # Loading data
    print('\n----------   Loading data ----------')
    train_loader, val_loader, test_loader = load_data(batch_size=128)
    print('\n----------   loading data is finished ----------')

    n_int = train_loader.dataset.size(1)
    # move network to gpu
    encoder = Encoder(n_int=n_int).to(device)
    decoder = Decoder(n_int=n_int).to(device)
    dynamics_learner = LatentODE().to(device)

    # dynamics_learner.load_state_dict(torch.load('Parameters_dynamics.pickle'))
    # encoder.load_state_dict(torch.load('Parameters_encoder.pickle'))
    # decoder.load_state_dict(torch.load('Parameters_decoder.pickle'))

    # Adam optimizer and the learning rate is 1e-3
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(dynamics_learner.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1)

    # Initialize the best validation error and corresponding epoch
    best_val_loss = np.inf
    best_epoch = 0

    loss_out = []
    loss1_out = []
    loss2_out = []
    loss3_out = []
    loss4_out = []

    print('\n----------   begin training  ----------')
    print('\n--   You need to wait about 10 minutes for each sub-epoch ')
    for epoch in range(1, args.epochs + 1):
        print(device)
        time1 = time.time()
        print('\n----------   Epoch %d/%d ----------' % (epoch, args.epochs))
        out_loss, loss1, loss2, loss3, loss4 = train_dynamics(args, encoder, decoder, dynamics_learner, optimizer,
                                                              scheduler, device, train_loader, use_cuda)
        val_loss = val_dynamics(args, encoder, decoder, dynamics_learner, device, val_loader, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        print('\nCurrent best epoch: %d, best val loss: %f' % (best_epoch, best_val_loss))

        loss_out.extend(out_loss)
        loss1_out.extend(loss1)
        loss2_out.extend(loss2)
        loss3_out.extend(loss3)
        loss4_out.extend(loss4)
        time2 = time.time()
        print("it spends %d seconds for trainsing this epoch" % (time2 - time1))

    print('\nBest epoch: %d' % best_epoch)

    test(args, encoder, decoder, dynamics_learner, device, test_loader)

    plt.plot(loss_out)
    plt.plot(loss1_out)
    plt.plot(loss2_out)
    plt.plot(loss3_out)
    plt.plot(loss4_out)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.yscale('log')
    plt.show()

    print('\n-----The code finishes running')

if __name__ == '__main__':
    main()
