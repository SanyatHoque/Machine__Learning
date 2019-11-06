clear all; close all; clc;
tic
x = load('ex4x.dat'); y = load('ex4y.dat');%[m, n] = size(x);
x = [ones(80, 1), x];
figure
plot(x((y == 1), 2), x((y == 1),3), '+');hold on
plot(x((y == 0), 2), x((y == 0), 3), 'o');hold on
xlabel('Exam 1 score')
ylabel('Exam 2 score')

theta = [0 0 0]';
% Newton's method
MAX_ITR = 10; J = [0 0 0 0 0 0 0]';
for i = 1:MAX_ITR
    %% Calculate the hypothesis function
    z = x * theta;
    h = (1.0 ./ (1.0 + exp(-z)));
    %h=g
    %% Calculate gradient and hessian.
    grad = (1/80).*x'* (h-y);
    H = (1/80).*x' * diag(h) * diag(1-h) * x;
    %% Calculate J (for testing convergence)
    J(i) =(1/80)*sum(-y.*log(h) - (1-y).*log(1-h));
    JJ=J(i);
    kk=(H\grad);
    %% Update theta
    theta = theta - (H\grad);
    theta1 =theta;
end
theta

z1 = [1 20 80] * theta;
h_g1 = (1.0 ./ (1.0 + exp(-z1)));
prob = 1 - h_g1

plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off
% Plot J
figure
plot(0:MAX_ITR-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')
% Display J
J;
toc