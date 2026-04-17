import torch
import numpy as np
from torch import nn
from torch.autograd.functional import jacobian
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import os
from scipy.linalg import eigh
import pickle
from scipy import io

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = {
    "font.family": 'Arial',  # 设置字体类型
    "font.size": 8,
}
rcParams.update(config)

def multivariate_normal_distribution(x, x0, sigma, n):
    inv_sigma = np.linalg.inv(sigma)
    exponent = np.exp(-0.5 * np.dot(np.dot((x - x0).T, inv_sigma), (x - x0)))
    coefficient = 1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5)
    return coefficient * exponent

def Cov_comp(sigmacell, circle, phi, N):
    Cov = np.zeros((N, N))
    mu = np.zeros(N)
    m = len(phi)
    for i in range(m):
        Cov += phi[i] * (sigmacell[:, :, i] + np.outer(circle[:, i], circle[:, i]))
        mu += phi[i] * circle[:, i]
    Cov -= np.outer(mu, mu)

    # 特征值分解与计算主成分
    D, V = eigh(Cov)
    rate1 = D[-1] / np.sum(D)
    rate2 = D[-2] / np.sum(D)
    V = V[:, np.argsort(D)[::-1]][:, :2]  # 取前两个最大的特征向量

    if np.dot(V[:, 0], np.ones(N)) < 0:
        V[:, 0] *= -1
    if np.dot(V[:, 1], np.ones(N)) < 0:
        V[:, 1] *= -1

    return rate1, rate2, V

def landscape(V, sigmacell, circle, phi, u1, u2, b1, b2):
    # 计算降维后的期望和方差（每个时间节点上）
    m = len(phi)
    sigma0_pca = []
    mu_pca = np.zeros((m, 2))
    for i in range(m):
        mu_pca[i, :] = np.dot(V.T, circle[:, i])
        sigma0_pca.append(np.dot(V.T, np.dot(sigmacell[:, :, i], V)))

    # 计算概率密度函数与landscape
    y_max = np.array([u1, u2])  # Range of the landscape
    y_min = np.array([b1, b2])
    step = (y_max - y_min) / 100  # Length of the step
    a1, a2 = np.meshgrid(np.arange(y_min[0], y_max[0] + step[0], step[0]),
                         np.arange(y_min[1], y_max[1] + step[1], step[1]))  # Grid
    s1, s2 = a1.shape
    P = np.zeros((s1, s2))  # 加权概率密度
    z = np.zeros((s1, s2))  # 单个概率密度
    for k in range(m):
        sig = sigma0_pca[k]
        x_wen = mu_pca[k]
        for i in range(s1):
            for j in range(s2):
                z[i, j] = multivariate_normal_distribution(np.array([a1[i, j], a2[i, j]]), x_wen, sig, 2)  # Normal distribution
        P += z * phi[k]

    G = np.sum(P)
    P /= G

    # Plot landscape
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(a1, a2, np.minimum(P, 1e-2),
                           alpha=1, rstride=1, cstride=1, cmap='jet')
    fig.colorbar(surf)
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_zlabel('U', fontsize=14)
    ax.set_xlim([b1, u1])
    ax.set_ylim([b2, u2])

    # # 绘制网格
    # for i in range(a1.shape[0] // 4):
    #     ax.plot(a1[4 * i, :], a2[4 * i, :], -np.log(np.maximum(P[4 * i, :], 1e-100)), color=[0.4, 0.4, 0.4],
    #             linewidth=0.01)
    # for i in range(a1.shape[1] // 4):
    #     ax.plot(a1[:, 4 * i], a2[:, 4 * i], -np.log(np.maximum(P[:, 4 * i], 1e-100)), color=[0.4, 0.4, 0.4],
    #             linewidth=0.01)

    # 绘制极限环
    X = np.dot(circle.T, V[:, 0])
    Y = np.dot(circle.T, V[:, 1])
    Z = np.zeros(m)
    for k in range(m):
        sig = sigma0_pca[k]
        x_wen = mu_pca[k]
        z = np.zeros(m)
        for i in range(m):
            z[i] = multivariate_normal_distribution(np.array([X[i], Y[i]]), x_wen, sig, 2)
        Z += phi[k] * z
    Z /= G
    Z = -np.log(np.maximum(Z, 1e-7))
    # ax.plot(X, Y, Z + 1, linewidth=10)

    plt.show()

    return X, Y, Z

class Encoder(nn.Module):
    def __init__(
        self,
        n_int: int,
        n_latent: int = 20,
        n_hidden: int = 32,
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
        n_hidden: int = 32,
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

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        return output

print('\n---------- Use this Use this part to infer the structure of GRNs ----------')
print('\n----------   import the DNN model  ----------')
time1 = time.time()
encoder = Encoder(n_int=44)
decoder = Decoder(n_int=44)
dynamics_learner = LatentODE()
dynamics_learner.load_state_dict(torch.load('Parameters_dynamics.pickle', map_location=torch.device('cpu')))
encoder.load_state_dict(torch.load('Parameters_encoder.pickle', map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load('Parameters_decoder.pickle', map_location=torch.device('cpu')))
dynamics_learner.eval()
encoder.eval()
decoder.eval()
print('\n----------   import DNN is finish  ----------')

## 提取Decoder的参数
fc1_weight = decoder.fc1.L1.weight.data
fc1_bias = decoder.fc1.L1.bias.data
fc2_weight = decoder.fc2.weight.data
fc2_bias = decoder.fc2.bias.data

## Decoder的雅可比矩阵
weight = torch.matmul(fc2_weight, fc1_weight).numpy()
bias = (torch.matmul(fc2_weight, fc1_bias) + fc2_bias).numpy()

# set initial points
with open('data_with_phase.pickle', 'rb') as f:
    adata = pickle.load(f)
np_data0 = adata[:, :-1]
np_data = (np_data0 - np_data0.min(0)) / (np_data0.max(0) - np_data0.min(0))
raw = torch.as_tensor(np_data, dtype=torch.float32)
latent, _ = encoder(raw)
latent_mean = latent.mean(0).detach()
latent_std = latent.std(0).detach()
x1 = torch.as_tensor(np_data[1, :].reshape(-1, 44), dtype=torch.float32)
encoded, _ = encoder(x1)
encoded = (encoded - latent_mean)/latent_std
sigma0 = np.eye(20)
latent = latent.detach().numpy()

new_weight = weight * latent.std(0)
new_bias = weight @ latent.mean(0) + bias

# the length of sequence we predict
NK = 10000

xx = np.zeros((20, NK+1))
sigma = np.zeros((20, 20, NK+1))

# get the original sequence
print('\n----------   get the original sequence  ----------')
xx[:,0] = encoded.detach().numpy()
sigma[:, :, 0] = sigma0

x1 = encoded

for i in range(NK):
    sigma1 = torch.as_tensor(sigma0, dtype=torch.float32)
    f = dynamics_learner(x1)
    # diag_jac = torch.zeros(20)
    # for j in range(20):
    #     x_perturbed = x1.clone()
    #     x_perturbed[0,j] += 1e-5
    #     f_perturbed = dynamics_learner(x_perturbed)
    #     diag_jac[j] = (f_perturbed[0,j] - f[0,j]) / 1e-5
    # diag_jac = diag_jac.detach().numpy()
    # f_np = f.detach().numpy()
    jac = jacobian(dynamics_learner, x1).squeeze()
    jac_np = jac.detach().numpy() # 转为numpy
    jac_diag = np.diag(jac_np)
    d_sigma = torch.matmul(sigma1, jac.t()) + torch.matmul(jac, sigma1) + 2 * 0.5 * torch.eye(20)  ## 这里改扩散系数D
    x1 = x1 + f * 0.005
    sigma0 = sigma1 + d_sigma * 0.005
    x0 = x1.detach().numpy()
    sigma0 = sigma0.detach().numpy()
    sigma0 = np.diag(np.diag(sigma0))
    xx[:, i+1] = x0
    sigma[:, :, i+1] = sigma0
    print('\n%d' % (i))

## 画图检查极限环
# tt = np.linspace(0, NK, NK+1)
# plt.plot(tt[2000:], sigma[2, 2, 2000:], linewidth=2.0)
# plt.show()

## 1674个格点达成一个周期
circle = xx[:, 9000:9499]
sigmacell= sigma[:, :, 9000:9499]
phi = np.ones(circle.shape[1]) / circle.shape[1]
N = 20

## 先在任意两维绘制landscape
V = np.zeros((N, 2))
V[8, 0] = 1
V[15, 1] = 1
u1 = 2.2
u2 = 2.2
b1 = -1.4
b2 = -2.1
# X, Y, Z = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2)

## 绘制降维后的landscape
# rate1, rate2, V = Cov_comp(sigmacell, circle, phi, N)
xreal = circle.T  # 取得一个周期的极限环
xpca = np.zeros((xreal.shape[0], 3))
xpca[:, 0] = 1
xpca[:, 1] = (xreal @ V)[:, 0]
xpca[:, 2] = (xreal @ V)[:, 1]
approx = np.linalg.lstsq(xpca, xreal, rcond=None)[0]
## 重构回原空间
testtemp = np.dot(xpca, approx)
r1 = np.zeros(N)
for i in range(N):
    temp1 = np.cov(testtemp[:, i], xreal[:, i])
    r1[i] = temp1[0, 1] / np.sqrt(temp1[0, 0] * temp1[1, 1])

## 计算投影后的F
# b1 = -4.3
# u1 = 5
# b2 = -4.5
# u2 = 4.2
y_max = np.array([u1, u2])
y_min = np.array([b1, b2])
step = 201
a1, a2 = np.meshgrid(np.linspace(y_min[0], y_max[0], step), np.linspace(y_min[1], y_max[1], step))
Fx_ols = np.zeros_like(a1)
Fy_ols = np.zeros_like(a1)
Fx_ols_norm = np.zeros_like(a1)
Fy_ols_norm = np.zeros_like(a1)

for i in range(a1.shape[0]):
    for j in range(a1.shape[1]):
        raw_point = np.array([1, a1[i, j], a2[i, j]]).dot(approx)
        x1 = torch.as_tensor(raw_point.reshape(1, -1), dtype=torch.float32)
        f = dynamics_learner(x1).detach().numpy()
        f_ols = f @ V
        Fx_ols[i, j] = f_ols[0][0]
        Fy_ols[i, j] = f_ols[0][1]
        norm = (f_ols[0][0] ** 2 + f_ols[0][1] ** 2) ** 0.5  # 归一化
        Fx_ols_norm[i, j] = f_ols[0][0] / norm
        Fy_ols_norm[i, j] = f_ols[0][1] / norm

## 计算极限环上的F
F = np.zeros_like(circle)
for i in range(circle.shape[1]):
    x1 = torch.as_tensor(circle[:, i].reshape(1, -1), dtype=torch.float32)
    f = dynamics_learner(x1)
    F[:, i] = f.detach().numpy()

# X, Y, Z = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2)

## 保存结果
# with open('landscape.pkl', 'wb') as f:
#     pickle.dump([sigmacell, circle, phi], f)

landscape_data = {
    'sigma': sigmacell,
    'circle': circle.T,
    'phi': phi.reshape(-1, 1),
    'weight': new_weight,
    'bias': new_bias,
    'a1': a1,
    'a2': a2,
    'Fx_ols': Fx_ols,
    'Fy_ols': Fy_ols,
    'Fx_ols_norm': Fx_ols_norm,
    'Fy_ols_norm': Fy_ols_norm,
    'F': F.T
}

io.savemat('landscape_nODE.mat', landscape_data)

time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))