close all, clear all, clc;
x=load('ex3x.dat'); y=load('ex3y.dat');
figure
plot3(x(:,1),x(:,2), y, 'o');
%whos % This shows the current variables
x = [ones(size(x,1),1) x];
sigma = std(x);
mu = mean(x);
x(:,2) = (x(:,2) - mu(:,2))./ sigma(:,2);
x(:,3) = (x(:,3) - mu(:,3))./ sigma(:,3);
trainingSamples = 47 ; 
numFeatures = 3 ;
theta = [0 0 0]';theta1 = [0 0 0]';theta2 = [0 0 0]';
maxIterations = 1500;
errorPerIteration = zeros(maxIterations,1);
for i=1:1500;
    totError = 0;totError1 = 0;totError2 = 0;
for j=1:numFeatures  
    df = 0; df1 = 0; df2 = 0; 
for m=1:47
prevTheta=theta;prevTheta1=theta1;prevTheta2=theta2;
h=0;h1=0;h2=0;  
for k=1:3
    h=h+prevTheta(k)*x(m,k);h1=h1+prevTheta1(k)*x(m,k);h2=h2+prevTheta2(k)*x(m,k);
end
df = df + (h-y(m))*x(m,j);df1 = df1 + (h1-y(m))*x(m,j);df2 = df2 + (h2-y(m))*x(m,j);   
totError = (totError + (h-y(m))*(h-y(m)))/m;
totError1 = (totError1 + (h1-y(m))*(h1-y(m)))/m;
totError2 = (totError2 + (h2-y(m))*(h2-y(m)))/m;
end
theta(j)=theta(j)-0.07*(df/m);theta1(j)=theta1(j)-0.03*(df1/m);theta2(j)=theta2(j)-0.01*(df2/m);
end
errorPerIteration(i)=totError/j;
errorPerIteration1(i)=totError1/j;errorPerIteration2(i)=totError2/j;   
prevTheta=theta;
prevTheta1=theta1;prevTheta2=theta2;
end
theta = prevTheta 
resultant_y = x*prevTheta;
b = (1650 - mu(:,2))/ sigma(:,2) ; 
c = (3 - mu(:,3))/ sigma(:,3) ;
price = [1 b c]*prevTheta
%1650 square feet and 3 bedrooms  $293,081
hold on
plot3(x(:,3),x(:,2),x*theta,'r+')
legend('Training data', 'Linear regression')

figure
plot(0:49,errorPerIteration(1:50),'red');
xlabel('Iteration', 'fontsize', 18, 'fontname', 'Arial');
ylabel('Error', 'fontsize', 18, 'fontname', 'Arial');
title('Iteration vs Error', 'fontsize', 20, 'fontname', 'Arial');
hold on;
plot(0:49,errorPerIteration1(1:50),'blue'); plot(0:49,errorPerIteration2(1:50),'green');