clc;clear;

load landscape_nODE.mat;
sigmacell = cell(length(phi),1);
for i = 1:length(phi)
    sigmacell{i} = sigma(:,:,i);
end
%% 绘制landscape
% clf
N = 20;
V = eye(N);
% [rate1, rate2, V] = Cov_comp(sigmacell, circle, phi, N);
% V = zeros(N,2);
% V(9,1) = 1;
% V(10,2) = 1;
F_proj = F * V;
circle_proj = circle * V;
D = 0.5; % 扩散系数
[Z, P, dP, flux] = landscape(V, sigmacell, circle, phi, F_proj, D);
flux_norm = sqrt(sum(flux.^2, 2)); % 计算flux的模长
% scatter(flux_norm,P);

% plot(Z);
step = 21;
% quiver(circle_proj(1:step:length(circle_proj),2),circle_proj(1:step:length(circle_proj),9), ...
%     flux(1:step:length(circle_proj),2),flux(1:step:length(circle_proj),9))

% 寻找检查点
% [~,l] = findpeaks(-Z);
% c1 = l(1); %M phase
% c2 = l(2); %G0/G1 phase
% c3 = l(3); %S/G2 phase
% l(4) = 425;
% c4 = l(4);
% basinpoint = circle(l,:);

% 计算flux integration
dl = diff([circle_proj;circle_proj(1,:)]); % 计算差分
dL = sqrt(sum(dl.^2, 2)); % 计算曲线长
Jdl = sum(flux .* dl, 2); % 计算flux与dl的内积
FPdl = sum(F_proj.*P.*dl, 2);
DdPdl = D*sum(dP.*dl, 2);
Flux_loop = sum(Jdl)/sum(dL); % flux integration
%% 用到的函数
% 计算分布的总体期望和方差并进行特征分解
function [rate1, rate2, V] = Cov_comp(sigmacell, circle, phi, N)
Cov = zeros(N,N);
mu = zeros(1,N);
m = length(phi);
for i = 1:m
    Cov = Cov + phi(i) * (sigmacell{i} + circle(i,:)'*circle(i,:));
    mu = mu + phi(i) * circle(i,:);
end
Cov = Cov - mu' * mu;
[~,D] = eigs(Cov,N);
rate1 = D(1,1) / sum(sum(D));
rate2 = D(2,2) / sum(sum(D));
[V,~] = eigs(Cov,2);
if sign(V(:,1)'*ones(N,1))<0
    V(:,1)=-V(:,1);
end
if sign(V(:,2)'*ones(N,1))<0
    V(:,2)=-V(:,2);
end
end

% 计算landscape（N维）；返回Z是极限环的landscape
function [Z, P, dP, flux] = landscape(V, sigmacell, circle, phi, F, D)
% V是坐标轴；circle是期望；phi是权重；sigmacell是方差；F是极限环上的向量场；D是扩散系数

% 计算降维后的期望和方差（每个时间节点上）
m = length(phi);
[~,n] = size(V);
sigma0_pca = cell(m,1);
mu_pca = zeros(m,n);
for i=1:m
   mu_pca(i,:) = V'*circle(i,:)';
   sigma0_pca{i} = V'*sigmacell{i}*V;
end

% 计算概率密度函数与landscape
P=zeros(m,1);%加权概率密度
z=zeros(m,1);%单个概率密度
dP=zeros(m,n);% P的偏导

for k=1:m  % 轨迹上以每个点为中心的高斯分布
    sig=sigma0_pca{k};
    x_wen=mu_pca(k,:);
    inv_sig=inv(sig);  %协方差矩阵的逆
    for i=1:m
        z(i)=multivariate_normal_distribution(mu_pca(i,:)',x_wen',sig,n);  %% Normal distribution
        gradient_z = z(i) * inv_sig * (x_wen' - mu_pca(i, :)'); % z的偏导数（由正态分布密度函数得到）
        dP(i,:) = dP(i,:) + phi(k) * gradient_z';
    end
    P = P + z * phi(k);
end
% G = sum(sum(P));
% P = P / G;
Z = -log(P);
flux = F.*P - D*dP; % 计算flux
end


% 多元正态分布密度函数
function z=multivariate_normal_distribution(x,x0,sigma,n)
z=1/((2*pi)^(n/2)*det(sigma)^(1/2))*exp(-0.5*(x-x0)'*(sigma)^(-1)*(x-x0));
end