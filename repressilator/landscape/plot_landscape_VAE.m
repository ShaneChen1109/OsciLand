clc;clear;

load landscape_VAE_nODE.mat;
sigmacell = cell(length(phi),1);
for i = 1:length(phi)
    sigmacell{i} = sigma(:,:,i);
end

V_pca = [
0.325587795805757	-0.381485300664108
-0.493356414825170	-0.0911433237176892
0.167720836769935	0.472832209599350
0.0772708919275938	-0.639553367721437
-0.592460176235521	0.253010839982375
0.515249361884211	0.386715096901951
];
%% 绘制landscape
clf
N = 5;
% [rate1, rate2, V] = Cov_comp(sigmacell, circle, phi, N);
V = zeros(N,2);
V(5,1) = 1;
V(6,2) = 1;
index1 = 5; index2 = 6;
weight_proj = weight([index1,index2],:)';
bias_proj = bias([index1,index2]);
% weight_proj = weight' * V_pca;
% bias_proj = bias * V_pca;
circle_proj = circle * weight_proj + bias_proj;

% b1 = -6.5; u1 = 6; b2 = -6.5; u2 = 6; 
b1 = 0; u1 = 8; b2 = 0; u2 = 8;
D = 0.001; % 扩散系数
[Z, U] = landscape(weight_proj, bias_proj, sigmacell, circle, phi, u1, u2, b1, b2, D);
l=[314;894;1486];
basin_point_latent = circle(l,:);
basin_point = basin_point_latent * weight'+ bias;
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
function [Z, U] = landscape(V, bias, sigmacell, circle, phi, u1, u2, b1, b2, D)
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
grid=300;
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
U = -log(max(P,10^-6));

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
% for i=1:2:floor(size(a1,1)/4)
%     plot3(a1(4*i-1,:),a2(4*i-1,:),-log(max(P(4*i-1,:),10^-5)),'Color',[0.4 0.4 0.4],'LineWidth',0.01);
% end
% for i=1:2:floor(size(a1,2)/4)
%     plot3(a1(:,4*i-1),a2(:,4*i-1),-log(max(P(:,4*i-1),10^-5)),'Color',[0.4 0.4 0.4],'LineWidth',0.01);
% end

% 绘制极限环
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
Z = -log(max(Z,10^-6));
% plot3(X,Y,Z+0.05,'r','lineWidth',2);
end


% 多元正态分布密度函数
function z=multivariate_normal_distribution(x,x0,sigma,n)
z=1/((2*pi)^(n/2)*det(sigma)^(1/2))*exp(-0.5*(x-x0)'*(sigma)^(-1)*(x-x0));
end