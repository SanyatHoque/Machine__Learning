%%(e),(f)
clear all; close all; clc;
data0 = load('twofeature.txt');
data = data0(1:end-1,:);
r = [randi([+1 +1],1,1000);randi([38 42],1,1000);randi([48 52],1,1000)]';
% data = [data;r];
x = data(:, 1);
y = data(:,2:3);
C = 1*10^20;      % C value is close to infinity, Acts as a Regularization Term
tic
model = svmtrain(y,x,'ShowPlot',true, 'boxconstraint',C,'autoscale', false);
w = model.Alpha'*model.SupportVectors;
b = model.Bias;   %16
toc
theta_SVM = [w b]
plot_x = [min(y(:,1)) max(y(:,1))];
plot_y = (-1/w(2))*(w(1)*plot_x + b);
hold on;
plot(plot_x, plot_y,'k','linewidth',5); %Line plotted by W and b
title(sprintf('SVM Linear Classifier'), 'FontSize', 14)

% clear all; close all; clc;
data000 = load('twofeature.txt');
data111 = data000(1:end-1,:); m = 50;
yy = data111(:, 1);
xx = data111(:,2:3);
for ii = 1:m
    if yy(ii)<0
       yy(ii) = 0;
    end
end
% neg = find(y == -1);
% x = [x(neg,2), x(neg,3)]
% x = load('ex4x.dat'); y = load('ex4y.dat');%[m, n] = size(x);
m = 50;
xx = [ones(m, 1), xx];
% plot(xx((yy == 1), 2), xx((yy == 1),3), '+');hold on
% plot(xx((yy == 0), 2), xx((yy == 0), 3), 'o');hold on
theta = [0 0 0]';
for i = 1:10
    z = xx * theta;
    h = (1.0 ./ (1 + exp(-z)));
    grad = (1/m).*xx'* (h-yy);
    H = (1/m).*xx' * diag(h) * diag(1-h) * xx;
    Error(i) =(1/m)*sum(-yy.*log(h) - (1-yy).*log(1-h));
    theta = theta - (H\grad);
end
% theta
% z1 = [1 20 80] * theta;
% h_g1 = (1.0 ./ (1.0 + exp(-z1)));
% prob = 1 - h_g1
% plot_x = [min(xx(:,2)),  max(xx(:,2))];
% plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
% plot(plot_x, plot_y)
hold on;
%%
% clear all; close all; clc;
data00 = load('twofeature.txt');
data1 = data00(1:end-1,:);
% r = [randi([+1 +1],1,1000);randi([38 42],1,1000);randi([48 52],1,1000)]';
% data = [data;r];
x = data1(:, 1);
y = data1(:,2:3);
C = 1;      %C = 0.009;
% tic
% model = svmtrain(y,x,'ShowPlot',true, 'boxconstraint',C,'autoscale', true);
% w = model.Alpha'*model.SupportVectors;
% b = model.Bias;   %16
% toc
% theta_SVM = [w b]
% plot_x = [min(y(:,1)) max(y(:,1))];
% plot_y = (-1/w(2))*(w(1)*plot_x + b);
% hold on;
% plot(plot_x, plot_y, 'k-', 'LineWidth', 2);
% title(sprintf('SVM Linear Classifier'), 'FontSize', 14)
posY1 =0; posY2 =0; negY1 =0; negY2 =0; midX =0; midY = 0;
countA =0; countB=0;
for i = 1:50;
    if (x(i)>0);
        posY1 = posY1 + y(i,1);
        posY2 = posY2 + y(i,2);
        countA = countA + 1;
    else (x(i)<0);
        negY1 = negY1 + y(i,1);
        negY2 = negY2 + y(i,2);
        countB = countB + 1;
    end
end
mean_pos1 = posY1 / countA;
mean_pos2 = posY2 / countA;
mean_neg1 = negY1 / countB;
mean_neg2 = negY2 / countB;
% draw line joing the centers
% find the midpoint
midpos = ( mean_pos1 + mean_neg1 ) / 2;%Midpoint of Cluster 1
midneg = ( mean_pos2 + mean_neg2 ) / 2;%Midpoint of Cluster 2
hold on;
plot (mean_pos1, mean_pos2,'o');
hold on;
plot (mean_neg1, mean_neg2,'o');
hold on;
plot (midpos , midneg,'X','markersize',19);%Midpoint of 2 clusters
g1 = (mean_pos1 - mean_neg1)/(mean_pos2 - mean_neg2);
g2 = -1/g1;
plot_x1 = [min(y(:,1)) max(y(:,1))];
plot_y1 = g2.*plot_x1 - g2*midpos + midneg;
hold on;
plot(plot_x1, plot_y1, 'k-', 'LineWidth', 2);
% y - y1 = [(y2 - y1) / (x2 - x1)]·(x - x1),