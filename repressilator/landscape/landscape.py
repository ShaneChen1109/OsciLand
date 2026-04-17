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

    def forward(self, x):
        fc1 = self.layer1(x)
        fc2 = self.layer2(fc1)
        output = self.layer4(fc2)
        # return 0.1492083 * output
        return 0.149676 * output

print('\n---------- Use this Use this part to infer the structure of GRNs ----------')
print('\n----------   import the DNN model  ----------')
time1 = time.time()
dynamics_learner = FullyConnected()

dynamics_learner.load_state_dict(torch.load('Parameters_saved1_5.pickle'))
dynamics_learner.eval()
print('\n----------   import DNN is finish  ----------')

# set initial points
kk = 2
x0 = np.ones((1, 6))*kk
sigma0 = np.eye(6)

# the length of sequence we predict
NK = 10000

xx = np.zeros((6, NK+1))
sigma = np.zeros((6, 6, NK+1))

# get the original sequence
print('\n----------   get the original sequence  ----------')
xx[:,0] = x0
sigma[:, :, 0] = sigma0


for i in range(NK):
    x1 = torch.as_tensor(x0, dtype=torch.float32)
    sigma1 = torch.as_tensor(sigma0, dtype=torch.float32)
    f = dynamics_learner(x1)
    jac = jacobian(dynamics_learner, x1).squeeze()
    d_sigma = torch.matmul(sigma1, jac.t()) + torch.matmul(jac, sigma1) + 2 * 0.005 * torch.eye(6)  ## 这里改扩散系数D
    x0 = x1 + f * 0.01
    sigma0 = sigma1 + d_sigma * 0.01
    x0 = x0.detach().numpy()
    sigma0 = sigma0.detach().numpy()
    sigma0 = np.diag(np.diag(sigma0))
    xx[:, i+1] = x0
    sigma[:, :, i+1] = sigma0
    print('\n%d' % (i))

## 画图检查极限环
# tt = np.linspace(0, NK, NK+1)
# plt.plot(tt[1000:], sigma[0, 0, 1000:], linewidth=2.0)
# plt.show()

## 1674个格点达成一个周期
circle = xx[:, 8000:9674]
sigmacell= sigma[:, :, 8000:9674]

phi = np.ones(circle.shape[1]) / circle.shape[1]


landscape_data = {
    'sigma': sigmacell,
    'circle': circle.T,
    'phi': phi.reshape(-1, 1)
}

io.savemat('landscape_nODE.mat', landscape_data)

N = 6
## 先在任意两维绘制landscape
# V = np.zeros((6, 2))
# V[0, 0] = 1
# V[1, 1] = 1
# u1 = 8
# u2 = 8
# b1 = 0
# b2 = 0
# X, Y, Z = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2)


## 绘制降维后的landscape
rate1, rate2, V = Cov_comp(sigmacell, circle, phi, N)
# u1 = 6
# u2 = 4
# b1 = -1
# b2 = 0
# X, Y, Z = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2)

time2 = time.time()
print('\n-----The code finishes running, and cost %d seconds' % (time2-time1))