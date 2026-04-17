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
index1 = 115; index2 = 116;
weight_proj = weight([index1,index2],:)';
bias_proj = bias([index1,index2]);
F_proj = F * weight_proj;
circle_proj = circle * weight_proj + bias_proj;

b1 = 0; u1 = 1; b2 = 0; u2 = 1; 
% b1 = -7; u1 = 4; b2 = -5; u2 = 5;
D = 0.20; % 扩散系数
[Z, U] = landscape(weight_proj, bias_proj, sigmacell, circle, phi, u1, u2, b1, b2, D);


% 寻找检查点
[~,l1] = findpeaks(-Z);
[~,l2] = findpeaks(Z);
% Barrier_1 = Z(l2(2)) - Z(l1(1));
% Barrier_2 = Z(l2(3)) - Z(l1(2));
% Barrier_3 = Z(l2(1)) - Z(l1(3));
% c1 = l1(1); %M phase
% c2 = l1(2); %G0/G1 phase
% c3 = l1(3); %S/G2 phase
% plot3(circle(c1, :) * V(:, 1), circle(c1, :) * V(:, 2), Z(c1) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% plot3(circle(c2, :) * V(:, 1), circle(c2, :) * V(:, 2), Z(c2) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% plot3(circle(c3, :) * V(:, 1), circle(c3, :) * V(:, 2), Z(c3) + 0.1, '.', 'Color', [1, 0.5, 0], 'MarkerSize', 30);
% text(circle(c1,:) * V(:,1), circle(c1,:) * V(:,2), Z(c1)+10,'M phase','Interpreter','latex','FontSize', 10);
% text(circle(c2,:) * V(:,1), circle(c2,:) * V(:,2), Z(c2)+10,'G0/G1 phase','Interpreter','latex','FontSize', 10);
% text(circle(c3,:) * V(:,1), circle(c3,:) * V(:,2), Z(c3)+10,'S/G2 phase','Interpreter','latex','FontSize', 10);

% exportgraphics(gcf, 'landscape_gene.pdf');
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

% surf(a1,a2,-log(max(P,10^-5)));   %% Plot landscape
pcolor(a1,a2,-log(max(P,10^-5)))
shading interp
xlim([b1 u1]);
ylim([b2 u2]);
% zlim([8 11.6]);
% colorbar
xlabel('CCNA2', 'FontSize', 40, 'FontName', 'Arial');
ylabel('CCNB1', 'FontSize', 40, 'FontName', 'Arial');
% zlabel('U', 'FontSize', 16, 'FontName', 'Arial');
set(gca, ...
    'XTick', linspace(b1, u1, 3), ...   
    'YTick', linspace(b2, u2, 3), ...  
    'TickLength', [0.015 0.015], ...
    'TickDir', 'out', ...               
    'LineWidth', 1.5, ...                 
    'Box', 'off', ...                   
    'FontSize', 20, ...
    'FontName', 'Arial');
% view([-65 77])
view(2); % 用于平面视图
set(gcf,'outerposition', [100 100 500 500])
set(gcf, 'PaperPositionMode', 'auto')
print(gcf, 'landscape_gene.pdf', '-dpdf', '-painters')

% hold on

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
Z = -log(max(Z,10^-5));
% plot3(X,Y,Z+0.05,'r','lineWidth',2);
end


% 多元正态分布密度函数
function z=multivariate_normal_distribution(x,x0,sigma,n)
z=1/((2*pi)^(n/2)*det(sigma)^(1/2))*exp(-0.5*(x-x0)'*(sigma)^(-1)*(x-x0));
end