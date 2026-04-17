clc;clear;

load landscape_nODE.mat;
sigmacell = cell(length(phi),1);
for i = 1:length(phi)
    sigmacell{i} = sigma(:,:,i);
end
%% 绘制landscape
clf
N = 20;
% [rate1, rate2, V] = Cov_comp(sigmacell, circle, phi, N);
% V = zeros(N,2);
% V(3,1) = 1;
% V(5,2) = 1;
index1 = 15; index2 = 22;
weight_proj = weight([index1,index2],:)';
bias_proj = bias([index1,index2]);
circle_proj = circle * weight_proj + bias_proj;

% b1 = -6; u1 = 7; b2 = -6; u2 = 11; 
b1 = 0; u1 = 1; b2 = 0; u2 = 1;

[Z, U] = landscape(weight_proj, bias_proj, sigmacell, circle, phi, u1, u2, b1, b2);

% 寻找检查点
[~,l1] = findpeaks(-Z);
[~,l2] = findpeaks(Z);
% plot3(circle_proj(l1, 1), circle_proj(l1, 2), Z(l1) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% plot3(circle_proj(l2, 1), circle_proj(l2, 2), Z(l2) + 0.1, '.', 'Color', [1.0, 0.6, 0.6], 'MarkerSize', 30);

exportgraphics(gcf, 'landscape.png', 'Resolution', 300);
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

% 绘制landscape（2维）；返回Z是极限环的landscape
function [Z, U] = landscape(V, bias, sigmacell, circle, phi, u1, u2, b1, b2)
% V是坐标轴；circle是期望；phi是权重；sigmacell是方差；u1,u2,b1,b2是landscape的范围；D是扩散系数

% 计算降维后的期望和方差（每个时间节点上）
m = length(phi);
sigma0_pca = cell(m,1);
mu_pca = zeros(m,2);
for i=1:m
   mu_pca(i,:) = V'*circle(i,:)';
   sigma0_pca{i} = V'*sigmacell{i}*V;
end
mu_pca = mu_pca + bias;

% 计算概率密度函数与landscape
y_max=[u1,u2]; %% Range of the landscape
y_min=[b1,b2];
grid=200;
step=(y_max-y_min)/grid; %% Length of the step
[a1,a2]=meshgrid(y_min(1):step(1):y_max(1),y_min(2):step(2):y_max(2)); %% Grid
[s1,s2]=size(a1);
P=zeros(s1,s2);%加权概率密度
z=zeros(s1,s2);%单个概率密度
for k=1:m
    sig=sigma0_pca{k};
    x_wen=mu_pca(k,:);
    for i=1:s1
        for j=1:s2
            z(i,j)=multivariate_normal_distribution([a1(i,j);a2(i,j)],x_wen',sig,2);  %% Normal distribution
        end
    end
    P = P + z * phi(k);
end
G = sum(sum(P));
P = P / G;
U = -log(P);

surf(a1,a2,-log(max(P,5*10^-6)));   %% Plot landscape
shading interp
xlim([b1 u1]);
ylim([b2 u2]);
% zlim([8.5 12.3]);
% colorbar
xlabel('CycE', 'FontSize', 40, 'FontName', 'Arial')
ylabel('CycA', 'FontSize', 40, 'FontName', 'Arial')
zlabel('U', 'FontSize', 40, 'FontName', 'Arial')
set(gca, 'FontSize', 20, 'FontName', 'Arial');
set(gca, 'LineWidth', 1.5);
view([-60 75])
% view(2); % 用于平面视图
set(gcf,'outerposition', [100 100 600 500])
set(gcf, 'PaperPositionMode', 'auto')
% saveas(gcf, 'landscape.png');

hold on

%% 绘制网格
for i=1:2:floor(size(a1,1)/5)
    plot3(a1(5*i-1,:),a2(5*i-1,:),-log(max(P(5*i-1,:),5*10^-6)),'Color',[0.4 0.4 0.4],'LineWidth',0.5);
end
for i=1:2:floor(size(a1,2)/5)
    plot3(a1(:,5*i-1),a2(:,5*i-1),-log(max(P(:,5*i-1),5*10^-6)),'Color',[0.4 0.4 0.4],'LineWidth',0.5);
end

%% 绘制极限环
X = circle * V(:,1) + bias(1);
Y = circle * V(:,2) + bias(2);
Z = zeros(m,1);
for k = 1:m
  sig = sigma0_pca{k};
  x_wen = mu_pca(k,:);
  z = zeros(m,1);
  for i = 1:m
  z(i) = multivariate_normal_distribution([X(i);Y(i)],x_wen',sig,2);
  end
  Z = Z + phi(k) * z; 
end
Z = Z / G;
Z = -log(max(Z,5*10^-6));
X = [X;X(1,:)]; % 用于让轨迹首位连接
Y = [Y;Y(1,:)];
Z1 = [Z;Z(1,:)];
plot3(X,Y,Z1+0.05, 'Color', [0.8, 0.4, 0.8],'lineWidth',2);
end


% 多元正态分布密度函数
function z=multivariate_normal_distribution(x,x0,sigma,n)
z=1/((2*pi)^(n/2)*det(sigma)^(1/2))*exp(-0.5*(x-x0)'*(sigma)^(-1)*(x-x0));
end