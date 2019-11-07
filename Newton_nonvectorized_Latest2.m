clear all; close all; clc;

x = load('ex4x.dat'); y = load('ex4y.dat');%[m, n] = size(x);
x = [ones(80, 1), x];
figure
plot(x((y == 1), 2), x((y == 1),3), '+');hold on
plot(x((y == 0), 2), x((y == 0), 3), 'o');hold on
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% x1 = x'*x;
%% Calculate Hessian Matrix x'x
x5 = [0,0,0;0,0,0;0,0,0];
for u = 1:3
    for w = 1:3
        for g = 1:80
            x5(w,u) = x5(w,u) + x(g,u)*x(g,w);
        end
    end
end

%% Initialize
theta = [0 0 0];
%max_iterations = 150000;
prevtheta = [0 0 0];
max_iterations = 400;
Cost_fun1 = zeros(max_iterations:1);
tolx = 0.4094;
tic
for i = 1:max_iterations
    gradient =[0 0 0];
    for j = 1:3
        Hessian = 0;
        Hessian1=[0,0,0;0,0,0;0,0,0];
        Cost_fun=0;
        for m = 1:80
            %% Calculate Hypothesis
            z=0;
            for k = 1:3
                z = z +  (prevtheta(k)*x(m,k));
            end
            %% Calculate gradient and hessian.
            h(m) =  (1 / (1 + exp(-z)));
            gradient(j) = gradient(j) + ((h(m)-y(m))*x(m,j));
            Hessian = Hessian + (h(m) * (1-h(m))) ;
            Cost_fun = Cost_fun + ((-y(m)*log(h(m))) - (1-y(m))*log(1-h(m)));
            
        end
        gradient(j) = (1/80)*gradient(j);
        Hessian = (1/80)*Hessian;
        Hessian1 = Hessian.*x5;
        Hessian2 = inv(Hessian1);
        Cost_fun1(i) = (1/80)*Cost_fun;
        if (Cost_fun1(i) < tolx)
            break
        end
        %% Calculate Newton's method, = grad/Hessian
        newton=[0,0,0];
        for u = 1:3
            for w = 1:3
                newton(u) = newton(u) + (Hessian2(w,u)*gradient(w));
            end
        end
        %%
        theta(j) = theta(j) - newton(j);
        
    end
    prevtheta =theta;
   
end
theta=prevtheta;
%%
toc
z1 = [1 20 80] * theta';
g1 = (1.0 ./ (1.0 + exp(-z1)));h1=g1;
probability = 1-h1;

plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

% Plot J
figure
plot(1:max_iterations, Cost_fun1,'--')
xlabel('Iteration'); ylabel('J')
% Display Cost_fun
Cost_fun;
