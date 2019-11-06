close all, clear all, clc;
x=load('ex2x.dat');
y=load('ex2y.dat');
figure
plot(x, y, 'o');

x = [ones(size(x,1),1) x];
trainingSamples = 50; 
numFeatures = 2 ;
maxIterations = 1500;
errorPerIteration = zeros(1500,1);
theta = [0 0]'; 


prevTheta=theta;
for i=1:1500

    
J = 0;    
for j=1:2
df = 0;
for m=1:50
h=0;
for k=1:2
h=h+prevTheta(k)*x(m,k); 
end
df = df + (h-y(m))*x(m,j);
J = J + (h-y(m))*(h-y(m)); 
end
J =J /m;
theta(j)=theta(j)-0.07*(df/m); 
end
prevTheta=theta;
errorPerIteration(i)=J/j; 

 
 
 end
theta = prevTheta
hold on
plot(x(:,2), x*theta, 'r')
legend('Training data', 'Linear regression')
height1 =  theta(1) + theta(2) * 3.5  
height2 =  theta(1) + theta(2) * 7    
%3.5 is 0.9737 meters, and for age 7 is 1.1975 meters.

J_vals = zeros(100, 100);   % initialize Jvals to 100x100 matrix of 0's
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
for i = 1:length(theta0_vals)
 for j = 1:length(theta1_vals)
	  J_vals(i,j) = (0.5/m).*(x*[theta0_vals(i); theta1_vals(j)]-y)'*(x*[theta0_vals(i); theta1_vals(j)]-y);
    end
end

% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals' ;
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')

figure;
% Plot the cost function with 15 contours spaced logarithmically
% between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 2, 15))
xlabel('\theta_0'); ylabel('\theta_1')
