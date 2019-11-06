close all, clear all, clc;
z =load('d2noisy.txt');
x1 = z(:,1:2);
y = z(:,3);
figure
plot3(x1(:,1),x1(:,2), y, 'o');

x = [ones(size(x1,1),1) x1];

trainingSamples = 50 ;%//number of rows in x 
numFeatures = 3 ;%//number of cols in x or number of constants
theta = [0 0 0]';
maxIterations = 500000; %500000 iterations, error 2.4057
errorPerIteration = zeros(500000,1);
for i=1:500000
    Error = 0;
for j=1:numFeatures  
    df = 0; df1 = 0; df2 = 0; 
for m=1:50
prevTheta=theta;
h=0;h1=0;h2=0;  
for k=1:2
    h=h+prevTheta(k)*x(m,k);
end
df = df + (h-y(m))*x(m,j);   
Error = (Error + (h-y(m))*(h-y(m)))/m;
end
theta(j)=theta(j)-0.07*(df/m);
end
errorPerIteration(i)=Error/j  ; 
prevTheta=theta;
end
hold on
theta=prevTheta
Final_Iteration_500000th = errorPerIteration(500000)
plot3(x1(:,1),x1(:,2),x*theta, 'r+')
legend('Training data', 'Linear regression')

figure
plot(0:49,errorPerIteration(1:50));
xlabel('Iteration', 'fontsize', 18, 'fontname', 'Arial');
ylabel('Error', 'fontsize', 18, 'fontname', 'Arial');
title('Iteration vs Error', 'fontsize', 20, 'fontname', 'Arial');
hold on;
%plot(0:49,errorPerIteration1(1:50)); plot(0:49,errorPerIteration2(1:50));