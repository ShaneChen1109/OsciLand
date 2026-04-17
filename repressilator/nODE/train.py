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
    print('\n Val data size:' , val_data.shape)
    print('\n Test data size:' , test_data.shape)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_dynamics(args, dynamics_learner, optimizer, scheduler, device, train_loader, use_cuda):

    dynamics_learner.train()
    loss_records = []

    # train sub_epochs times before every validation
    for step in range(1, args.sub_epochs + 1):
        time3 = time.time()
        loss_record = []
        mse_record = []
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            loss, mse = train_dynamics_learner(optimizer, dynamics_learner, data, args.prediction_steps, use_cuda,
                                               device)
            loss_record.append(loss.item())
            mse_record.append(mse.item())

        loss_out = np.sum(loss_record) / len(train_loader.dataset)
        mse_out = np.sum(mse_record) / len(train_loader.dataset)
        loss_records.append(loss_out)
        current_lr = optimizer.param_groups[0]['lr']
        print('\nTraining %d/%d before validation, loss: %f, MSE: %f' % (step, args.sub_epochs, loss_out, mse_out))
        print(f'Current learning rate: {current_lr}')
        scheduler.step()
        time4 = time.time()
        print("it spends %d seconds for trainsing this sub-epoch" % (time4-time3))

    return loss_records


def val_dynamics(args, dynamics_learner, device, val_loader, best_val_loss):

    dynamics_learner.eval()
    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, data, args.prediction_steps, device)
        loss_record.append(loss.item())
        mse_record.append(mse.item())

    loss_out = np.sum(loss_record) / len(val_loader.dataset)
    mse_out = np.sum(mse_record) / len(val_loader.dataset)
    print('\nValidation: loss: %f, MSE: %f' % (loss_out, mse_out))

    if best_val_loss > loss_out:
        torch.save(dynamics_learner.state_dict(), args.dynamics_path)

    return loss_out


def train_dynamics_learner(optimizer, dynamics_learner, data, steps, use_cuda, device):
    optimizer.zero_grad()

    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]

    # Make a prediction with steps-1，output：batchsize, num_nodes, time_steps, dimension
    x0 = torch.as_tensor(input1, dtype=torch.float32).view(-1, data.size(1)).to(device)
    t = torch.arange(0, 0.05 * steps, 0.05).to(device)
    solution = odeint(dynamics_learner, x0, t, method='rk4')
    outputs = solution[1:, :, :]
    outputs = outputs.permute(1, 2, 0).unsqueeze(-1)

    loss = F.l1_loss(outputs, target) * data.size(0)
    loss.backward()
    optimizer.step()
    mse = F.mse_loss(outputs, target) * data.size(0)
    if use_cuda:
        loss = loss.cpu()
        mse = mse.cpu()

    return loss, mse


def val_dynamics_learner(dynamics_learner, data, steps, device):

    input1 = data[:, :, 0, :]
    target = data[:, :, 1: steps, :]

    x0 = torch.as_tensor(input1, dtype=torch.float32).view(-1, data.size(1)).to(device)
    t = torch.arange(0, 0.05 * steps, 0.05).to(device)
    solution = odeint(dynamics_learner, x0, t, method='rk4')
    outputs = solution[1:, :, :]
    outputs = outputs.permute(1, 2, 0).unsqueeze(-1)

    loss = F.l1_loss(outputs, target) * data.size(0)
    mse = F.mse_loss(outputs, target) * data.size(0)

    return loss, mse


def test(args, dynamics_learner, device, test_loader):
    # load model
    dynamics_learner.load_state_dict(torch.load(args.dynamics_path))
    dynamics_learner.eval()
    loss_record = []
    mse_record = []
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        loss, mse = val_dynamics_learner(dynamics_learner, data, args.prediction_steps, device)
        loss_record.append(loss.item())
        mse_record.append(mse.item())

    loss_out = np.sum(loss_record) / len(test_loader.dataset)
    mse_out = np.sum(mse_record) / len(test_loader.dataset)
    print('loss: %f, mse: %f' % (loss_out, mse_out))


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=6, out_features=32, bias=True),
            nn.ELU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.BatchNorm1d(32),
            nn.ELU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=6, bias=True)
            )

    def forward(self, t, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        return output


def main():

    # Training settings

    parser = argparse.ArgumentParser(description='DNN_FOR_NETWORK_REFERENCE')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1998)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs, default: 5)')
    parser.add_argument('--sub-epochs', type=int, default=2,
                        help='i.e. train 10 times before every Validation (default: 10)')
    parser.add_argument('--prediction-steps', type=int, default=50,
                        help='prediction steps in data (default: 100)')
    parser.add_argument('--dynamics-path', type=str, default='Parameters_saved.pickle',
                        help='path to save dynamics learner (default: ./Parameters_saved.pickle)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device:', device)

    # torch.manual_seed(args.seed)

    # Loading data
    print('\n----------   Loading data ----------')
    train_loader, val_loader, test_loader = load_data(batch_size=128)
    print('\n----------   loading data is finished ----------')

    # move network to gpu
    dynamics_learner = FullyConnected().to(device)
    # Adam optimizer and the learning rate is 1e-3
    optimizer = torch.optim.Adam(dynamics_learner.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Initialize the best validation error and corresponding epoch
    best_val_loss = np.inf
    best_epoch = 0

    loss_out = []
    print('\n----------   Parameters of each layer  ----------')
    for name, parameters in dynamics_learner.named_parameters():
        print(name,":",parameters.shape)

    print('\n----------   begin training  ----------')
    print('\n--   You need to wait about 10 minutes for each sub-epoch ')
    for epoch in range(1, args.epochs + 1):
        print(device)
        time1 = time.time()
        print('\n----------   Epoch %d/%d ----------' % (epoch,args.epochs))
        out_loss = train_dynamics(args, dynamics_learner, optimizer, scheduler, device, train_loader, use_cuda)
        val_loss = val_dynamics(args, dynamics_learner, device, val_loader, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        print('\nCurrent best epoch: %d, best val loss: %f' % (best_epoch, best_val_loss))

        loss_out.extend(out_loss)
        time2 =time.time()
        print("it spends %d seconds for trainsing this epoch" % (time2-time1))

    print('\nBest epoch: %d' % best_epoch)

    test(args, dynamics_learner, device, test_loader)

    plt.plot(loss_out)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.yscale('log')
    plt.show()

    print('\n-----The code finishes running' )

if __name__ == '__main__':
    main()

