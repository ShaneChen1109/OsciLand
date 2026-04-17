clc;clear;

load landscape_nODE.mat;
load flux.mat;
load latent_z.mat;

sigmacell = cell(length(phi),1);
for i = 1:length(phi)
    sigmacell{i} = sigma(:,:,i);
end
% load result_for_landscape.mat;
%% 绘制landscape
clf
N = 20;
% [rate1, rate2, V] = Cov_comp(sigmacell, circle, phi, N);
V = zeros(N,2);
V(9,1) = 1;
V(16,2) = 1;
F_proj = F * V;
circle_proj = circle * V;
flux_proj = flux * V;

% 获得投影矩阵
[n0,~] = size(circle);
xlow = zeros(n0,3);
xlow(:,1) = 1;
xlow(:,2:3) = circle * V;
approx = xlow \ circle; % 等价于求解最小二乘，得到投影矩阵
testtemp = xlow * approx; % 重构回原空间
r_test = zeros(N,1); % 评估拟合效果
for i = 1:N
    temp = cov(squeeze(testtemp(:,i)),squeeze(circle(:,i)));
    r_test(i) = temp(1,2) / sqrt(temp(1,1) * temp(2,2));
end

b1 = -1.4; u1 = 2.2; b2 = -2.1; u2 = 2.2; 
% b1 = -4.3; u1 = 5; b2 = -4.5; u2 = 4.2;
D = 0.5; % 扩散系数
scatters = latent_z(:,[9,16,21]); % 绘制数据点
% scatters = zeros(8000,3);
% scatters(:,1:2) = latent_z(:,1:20) * V;
% scatters(:,3) = latent_z(:,21);
[Z, U] = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2, D, Fx_ols, Fy_ols, scatters);

% 找极大值点和极小值点
% [row, col] = find(imregionalmax(U));
% [m, n] = size(U);
% isNotOnBoundary = (row > 1) & (row < m) & (col > 1) & (col < n);
% row = row(isNotOnBoundary);
% col = col(isNotOnBoundary);
% max_U = U(row, col);
% min_U = min(U(:));
% Barrier_center = max_U - min_U;
% plot3(a1(row,col), a2(row,col), U(row, col), 'r*', 'MarkerSize', 10);

% 寻找检查点
[~,l1] = findpeaks(-Z);
[~,l2] = findpeaks(Z);
% Barrier_1 = Z(l2(2)) - Z(l1(1));
% Barrier_2 = Z(l2(3)) - Z(l1(2));
% Barrier_3 = Z(l2(1)) - Z(l1(3));
% c1 = l1(1); %M phase
% c2 = l1(2); %G0/G1 phase
% c3 = l1(3); %S/G2 phase
% plot3(circle(l1, :) * V(:, 1), circle(l1, :) * V(:, 2), Z(l1) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% plot3(circle(l2, :) * V(:, 1), circle(l2, :) * V(:, 2), Z(l2) + 0.1, '.', 'Color', [1.0, 0.6, 0.6], 'MarkerSize', 30);
% plot3(circle(c1, :) * V(:, 1), circle(c1, :) * V(:, 2), Z(c1) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% plot3(circle(c2, :) * V(:, 1), circle(c2, :) * V(:, 2), Z(c2) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% plot3(circle(c3, :) * V(:, 1), circle(c3, :) * V(:, 2), Z(c3) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% text(circle(c1,:) * V(:,1), circle(c1,:) * V(:,2), Z(c1)+10,'M phase','Interpreter','latex','FontSize', 10);
% text(circle(c2,:) * V(:,1), circle(c2,:) * V(:,2), Z(c2)+10,'G0/G1 phase','Interpreter','latex','FontSize', 10);
% text(circle(c3,:) * V(:,1), circle(c3,:) * V(:,2), Z(c3)+10,'S/G2 phase','Interpreter','latex','FontSize', 10);

step=29; % 504的因数
magnitudes = vecnorm(flux_proj, 2, 2);
flux_norm = flux_proj./magnitudes;
% quiver3(circle_proj(1:step:length(circle_proj),1),circle_proj(1:step:length(circle_proj),2),Z(1:step:length(circle_proj),1)+2, ...
%     flux_norm(1:step:length(circle_proj),1),flux_norm(1:step:length(circle_proj),2),0*Z(1:step:length(circle_proj),1), ...
%     'Color','w','LineWidth',1.5,'AutoScale', 'on', 'AutoScaleFactor', 0.5);

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
function [Z, U] = landscape(V, sigmacell, circle, phi, u1, u2, b1, b2, D, Fx, Fy, scatters)
% V是坐标轴；circle是期望；phi是权重；sigmacell是方差；u1,u2,b1,b2是landscape的范围；D是扩散系数

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
grid=200;
step=(y_max-y_min)/grid; %% Length of the step
dx=step(1); dy=step(2);
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

[GUx,GUy] = gradient(U,dx,dy);
[GPx,GPy] = gradient(P,dx,dy);

EEE=Fx.^2+Fy.^2; % 计算F
FFx=Fx./(sqrt(EEE)+eps);
FFy=Fy./(sqrt(EEE)+eps);

Fgradx = -D*GUx; % 计算负梯度力
Fgrady = -D*GUy;
EE=Fgradx.^2+Fgrady.^2;
FFgradx=Fgradx./(sqrt(EE)+eps);
FFgrady=Fgrady./(sqrt(EE)+eps);

Jx = Fx.*P - D*GPx; % 计算flux
Jy = Fy.*P - D*GPy;
E=Jy.^2+Jx.^2;
JJx=Jx./(sqrt(E));
JJy=Jy./(sqrt(E));

Fcx = Fx - Fgradx; % 计算旋度力
Fcy = Fy - Fgrady;
EEEE=Fcx.^2+Fcy.^2;
FFcx=Fcx./(sqrt(EEEE)+eps);
FFcy=Fcy./(sqrt(EEEE)+eps);

surf(a1,a2,-log(max(P,10^-5)));   %% Plot landscape
% pcolor(a1,a2,-log(max(P,10^-5)))
shading interp
xlim([b1 u1]);
ylim([b2 u2]);
% zlim([9.6 11.52]);
% colorbar
xlabel('latent9', 'FontSize', 40, 'FontName', 'Arial');
ylabel('latent16', 'FontSize', 40, 'FontName', 'Arial');
zlabel('U', 'FontSize', 40, 'FontName', 'Arial');
set(gca, 'FontSize', 20, 'FontName', 'Arial');
% set(gca, 'LineWidth', 1.5);
% view([-45 75])
% view([-64 81])
view(2); % 用于平面视图
set(gcf,'outerposition', [100 100 500 500])
set(gcf, 'PaperPositionMode', 'auto')
% saveas(gcf, 'landscape.png');

hold on

%% 绘制F、flux和负梯度力
mg = 1:10:(grid+1);
ng = mg;

% quiver3(a1(mg,ng),a2(mg,ng),14*ones(21,21), ...
%     FFx(mg,ng),FFy(mg,ng),0*FFx(mg,ng), ...
%     0.45,'Color','k','LineWidth',1); % F

quiver3(a1(mg,ng),a2(mg,ng),14*ones(21,21), ...
    FFgradx(mg,ng),FFgrady(mg,ng),0*FFgradx(mg,ng), ...
    0.45,'Color','w','LineWidth',1); % 负梯度力

% quiver3(a1(mg,ng),a2(mg,ng),14*ones(21,21), ...
%     10*Jx(mg,ng),10*Jy(mg,ng),0*JJx(mg,ng), ...
%     2.0,'Color',[1,0.6,0.6],'LineWidth',1); % flux

quiver3(a1(mg,ng),a2(mg,ng),14*ones(21,21), ...
    FFcx(mg,ng),FFcy(mg,ng),0*FFcx(mg,ng), ...
    0.45,'Color',[1,0.6,0.6],'LineWidth',1); % 旋度力

%% 绘制网格
% for i=1:2:floor(size(a1,1)/5)
%     plot3(a1(5*i-1,:),a2(5*i-1,:),-log(max(P(5*i-1,:),10^-5)),'Color',[0.4 0.4 0.4],'LineWidth',0.5);
% end
% for i=1:2:floor(size(a1,2)/5)
%     plot3(a1(:,5*i-1),a2(:,5*i-1),-log(max(P(:,5*i-1),10^-5)),'Color',[0.4 0.4 0.4],'LineWidth',0.5);
% end

%% 绘制极限环
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
Z = -log(max(Z,10^-5));
X = [X;X(1,:)]; % 用于让轨迹首位连接
Y = [Y;Y(1,:)];
Z1 = [Z;Z(1,:)];
% plot3(X,Y,Z1+0.05, 'Color', [0.8, 0.4, 0.8],'lineWidth',2);

%% 绘制散点
n_s = length(scatters);
Z_s = zeros(n_s,1);
for k = 1:m
  sig = sigma0_pca{k};
  x_wen = mu_pca(k,:);
  z_s = zeros(n_s,1);
  for i = 1:n_s
  z_s(i) = multivariate_normal_distribution(scatters(i,1:2)',x_wen',sig,2);
  end
  Z_s = Z_s + phi(k) * z_s; 
end
Z_s = Z_s / G;
U_s = -log(max(Z_s,10^-5));

colors = [
    1.0, 0.4980392156862745, 0.054901960784313725;  % Phase 0
    0.8901960784313725, 0.4666666666666667, 0.7607843137254902;  % Phase 1
    0.8392156862745098, 0.15294117647058825, 0.1568627450980392;  % Phase 2
    0.5803921568627451, 0.403921568627451, 0.7411764705882353   % Phase 3
];
category_indices = scatters(:,3);

scatter_colors = colors(category_indices+1, :);

% scatter3(scatters(:,1), scatters(:,2),U_s+0.1, 5, scatter_colors, 'filled', 'MarkerFaceAlpha', 0.6);
% scatter3(scatters(:,1), scatters(:,2), Z_s, 20, scatter_colors, 'filled', 'MarkerFaceAlpha', 0.6);
end


% 多元正态分布密度函数
function z=multivariate_normal_distribution(x,x0,sigma,n)
z=1/((2*pi)^(n/2)*det(sigma)^(1/2))*exp(-0.5*(x-x0)'*(sigma)^(-1)*(x-x0));
end