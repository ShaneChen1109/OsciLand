clc;clear;
% start = [5,5,0,1,3,1];
% [~,y1] = ode45(@(t,x) of(x),[0 2000],start);
% start = y1(end,:);
% [t1,y1] = ode45(@(t,x) of(x),[0 1000],start);
% [M,l] = findpeaks(y1(:,1));
% clear y1;
% Temp = t1(l);
% Temp = diff(Temp);
% T = mean(Temp); % 计算振荡周期
% 
% D = 0.005; % 扩散系数
% n = 6;
% start_2 = [ones(1,n),start];
% [~,y2] = ode15s(@(t,x) [2 * Phi(x(n+1:2*n)).*x(1:n) + 2 * D ; of(x(n+1:2*n))] , [0 50], start_2);
% start_2 = y2(end,:);
% [~,y2] = ode15s(@(t,x) [2 * Phi(x(n+1:2*n)).*x(1:n) + 2 * D ; of(x(n+1:2*n))] , [0 50], start_2);
% start_2 = y2(end,:);
% [t2,y2] = ode15s(@(t,x) [2 * Phi(x(n+1:2*n)).*x(1:n) + 2 * D ; of(x(n+1:2*n))] , 0:0.01:T, start_2); % 计算一个周期内方程的解
% S1 = y2(:,1:n); % 方差
% circle = y2(:,n+1:2*n); % 期望
% t = t2;
% sigmacell = cell(length(t),1);% 协方差阵（对角）
% for i = 1:length(t)
%     sigmacell{i} = diag(S1(i,:));
% end
% phi = [diff(t);0.01];% 计算权重（认为密度关于时间平均）
% phi = phi / sum(phi);

% load result_for_landscape.mat;

load landscape_nODE.mat;
sigmacell = cell(length(phi),1);
for i = 1:length(phi)
    sigmacell{i} = sigma(:,:,i);
end
%% 绘制landscape
clf
N = 6;
% [rate1, rate2, V] = Cov_comp(sigmacell, circle, phi, N);
V = zeros(6,2);
V(5,1) = 1;
V(6,2) = 1;
% V = [
%     0.325587795805757	-0.381485300664108
%     -0.493356414825170	-0.0911433237176892
%     0.167720836769935	0.472832209599350
%     0.0772708919275938	-0.639553367721437
%     -0.592460176235521	0.253010839982375
%     0.515249361884211	0.386715096901951
% ];
% b1 = -6.5; u1 = 6; b2 = -6.5; u2 = 6; 

% load V.mat;
b1 = 0; u1 = 8; b2 = 0; u2 = 8;
Z = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2);

%% 用到的函数
% 微分方程右端项
function F=of(x)
    a=10;
    a0=1e-3*a;
    b=0.5;
    n=2;

    F=zeros(6,1);
    F(1) = -b * (x(1)-x(4));
    F(2) = -b * (x(2)-x(5));
    F(3) = -b * (x(3)-x(6));
    F(4) = -x(4) + a / (1 + x(3)^n) + a0;
    F(5) = -x(5) + a / (1 + x(1)^n) + a0;
    F(6) = -x(6) + a / (1 + x(2)^n) + a0;
    
end

% 雅克比矩阵
function result = Phi(t)
syms x [6 1]
a = 10; a0 = 1e-3 * a; b = 0.5; n = 2;
g = jacobian([-b * (x1-x4);
    -b * (x2-x5);
    -b * (x3-x6);
    -x4 + a / (1 + x3^n) + a0;
    -x5 + a / (1 + x1^n) + a0;
    -x6 + a / (1 + x2^n) + a0]);
g = subs(g,{'x1','x2','x3','x4','x5','x6',},{t(1),t(2),t(3),t(4),t(5),t(6)});
g = double(g);
g = diag(g);
result = g;
end

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
[~,D] = eigs(Cov);
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
function Z = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2)
% V是坐标轴；circle是期望；phi是权重；sigmacell是方差；u1,u2,b1,b2是landscape的范围

% 计算降维后的期望和方差（每个时间节点上）
m = length(phi);
sigma0_pca = cell(m,1);
mu_pca = zeros(m,2);
for i=1:m
   mu_pca(i,:) = V'*circle(i,:)';
   sigma0_pca{i} = V'*sigmacell{i}*V;
end

% 计算概率密度函数与landscape
y_max=[u1,u2]; %% Range of the landscape
y_min=[b1,b2];
step=(y_max-y_min)/300; %% Length of the step
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
surf(a1,a2,(max(P,10^-6)));   %% Plot landscape
shading interp
xlim([b1 u1]);
ylim([b2 u2]);
zlim([0 6*10^-4]);
% xlabel('LacI (protein)', 'FontSize', 22, 'FontName', 'Arial');
% ylabel('CI (protein)', 'FontSize', 22, 'FontName', 'Arial');
xlabel('TetR', 'FontSize', 35, 'FontName', 'Arial');
ylabel('CI', 'FontSize', 35, 'FontName', 'Arial');
zlabel('P', 'FontSize', 35, 'FontName', 'Arial');
set(gca, 'FontSize', 30, 'FontName', 'Arial');

view([-25 75])
set(gcf, 'Position', [100, 100, 600, 500]);
set(gcf, 'PaperPositionMode', 'auto');
% colorbar
exportgraphics(gcf, 'landscape.png', 'Resolution', 300);

hold on

% 绘制网格
% for i=1:floor(size(a1,1)/4)
%     plot3(a1(4*i-1,:),a2(4*i-1,:),-log(max(P(4*i-1,:),10^-4)),'Color',[0.4 0.4 0.4],'LineWidth',0.01);
% end
% for i=1:floor(size(a1,2)/4)
%     plot3(a1(:,4*i-1),a2(:,4*i-1),-log(max(P(:,4*i-1),10^-4)),'Color',[0.4 0.4 0.4],'LineWidth',0.01);
% end

% 绘制极限环
X = circle * V(:,1);
Y = circle * V(:,2);
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
Z = -log(max(Z,10^-6));
% plot3(X,Y,Z+0.1,'lineWidth',1);
end

% 多元正态分布密度函数
function z=multivariate_normal_distribution(x,x0,sigma,n)
z=1/((2*pi)^(n/2)*det(sigma)^(1/2))*exp(1)^((-0.5)*(x-x0)'*(sigma)^(-1)*(x-x0));
end