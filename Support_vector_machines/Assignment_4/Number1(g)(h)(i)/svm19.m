%%(g),(h),(i)
clear all; close all; clc;
data0 = load('twofeature.txt');
data = data0(1:end-1,:);
r = [randi([+1 +1],1,1000);randi([38 42],1,1000);randi([48 52],1,1000)]';
data = [data;r];
x = data(:, 1);
y = data(:,2:3);
C = 1*10^20; %  C value is close to infinity, Acts as a Regularization Term
tic
model = svmtrain(y,x,'ShowPlot',true, 'boxconstraint',C,'autoscale', false);
w = model.Alpha'*model.SupportVectors;
b = model.Bias;   %16
toc
theta = [w b]
% Plot the data points
% figure
% pos = find(x == 1);
% neg = find(x == -1);
% plot(y(pos,1), y(pos,2), 'ko', 'MarkerFaceColor', 'b'); hold on;
% plot(y(neg,1), y(neg,2), 'ko', 'MarkerFaceColor', 'g')
% Plot the decision boundary
plot_x = [min(y(:,1)) max(y(:,1))];
plot_y = (-1/w(2))*(w(1)*plot_x + b);
hold on;
plot(plot_x, plot_y,'k','linewidth',5); %Line plotted by W and b
title(sprintf('SVM Linear Classifier with C = %g', C), 'FontSize', 14)
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
midpos = ( mean_pos1 + mean_neg1 ) / 2; %Midpoint of Cluster 1
midneg = ( mean_pos2 + mean_neg2 ) / 2; %Midpoint of Cluster 2
hold on;
plot (mean_pos1, mean_pos2,'o');
hold on;
plot (mean_neg1, mean_neg2,'o');
hold on;
plot (midpos , midneg,'X','markersize',19);%Midpoint of 2 clusters