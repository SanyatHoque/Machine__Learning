close all;clear all;clc;
 
x1 = [0;1;0;1;]; 
x2 = [0;0;1;1;];
y  = [1;0;0;1;]; 
% x1 = [0]; 
% x2 = [1];
% y  = [0]; 
%% Initilizing with Random Weights
data = randperm(20)-10;
W11_1 = data(1); W12_1 = data(2); b11_1 = data(3);  
W21_1 = data(4); W22_1 = data(5); b21_1 = data(6); 
W11_2 = data(7); W12_2 = data(8); b11_2 = data(9);
 
 
for i = 1:1000000     %1000000 
     m1 = randperm(4);
     for m = m1(1:4)
        %% Forward prop
        a1_2 = W11_1.*x1(m) + W12_1.*x2(m) + b11_1;%2nd Layer 1st Neuron 
        a2_2 = W21_1.*x1(m) + W22_1.*x2(m) + b21_1;%2nd Layer 2nd Neuron
        h1_2 = 1./(1+exp(-a1_2));%Hypothesis of 2nd Layer 1st Neuron
        h2_2 = 1./(1+exp(-a2_2));%Hypothesis of 2nd Layer 2nd Neuron
        a1_3 = W11_2.*h1_2 + W12_2.*h2_2 + b11_2;%Output Layer 1st Neuron
        h1_3(m) = 1./(1+exp(-a1_3));%Hypothesis of Output Layer 1st Neuron
        del1_3 = (h1_3(m) - y(m)); %Del function
        
        %% Back prop
        W11_2 = W11_2 - 0.007.*(h1_3(m) - y(m)).*h1_2;%Weight of 2nd Layer 1st Neuron
        W12_2 = W12_2 - 0.007.*(h1_3(m) - y(m)).*h2_2;%Weight of 2nd Layer 2nd Neuron
        b11_2 = b11_2 - 0.007.*(h1_3(m) - y(m));%Intercept Weight of 2nd Layer 
        
        W11_1 = W11_1 -0.007.*(h1_3(m) - y(m)).*W11_2.*h1_2.*(1-h1_2).*x1(m);%Weight of 1st Layer 1st Neuron
        W12_1 = W12_1 -0.007.*(h1_3(m) - y(m)).*W11_2.*h1_2.*(1-h1_2).*x2(m);%Weight of 1st Layer 2nd Neuron
        b11_1 = b11_1 -0.007.*(h1_3(m) - y(m)).*W11_2.*h1_2.*(1-h1_2);
        W21_1 = W21_1 -0.007.*(h1_3(m) - y(m)).*W12_2.*h2_2.*(1-h2_2).*x1(m);%Weight of 1st Layer 1st Neuron
        W22_1 = W22_1 -0.007.*(h1_3(m) - y(m)).*W12_2.*h2_2.*(1-h2_2).*x2(m);%Weight of 1st Layer 2nd Neuron
        b21_1 = b21_1 -0.007.*(h1_3(m) - y(m)).*W12_2.*h2_2.*(1-h2_2);
    end
     error1(i) = h1_3(1) - y(1);
     error2(i) = h1_3(2) - y(2);
     error3(i) = h1_3(3) - y(3);
     error4(i) = h1_3(4) - y(4);
end
%% Testing the Resuts
% After sufficient iterations, the features extracted are
% similar to slide 25 lecture notes for Intial random weights 
x1 = [0 1 0 1] 
x2 = [0 0 1 1]
for m = 1:4
    a1_2 = W11_1.*x1(m) + W12_1.*x2(m) + b11_1;
    a2_2 = W21_1.*x1(m) + W22_1.*x2(m) + b21_1;
    h1_2 = 1./(1+exp(-a1_2));
    h2_2 = 1./(1+exp(-a2_2));
    a1_3 = W11_2.*h1_2 + W12_2.*h2_2 + b11_2;
    h1_3(m) = 1./(1+exp(-a1_3));
end
h1_3


figure
plot(error1)
ylabel('error1stsample');xlabel('iterations')
figure
plot(error2)
ylabel('error2ndsample');xlabel('iterations')
figure
plot(error3)
ylabel('error3rdsample');xlabel('iterations')
figure
plot(error4)
ylabel('error4thsample');xlabel('iterations')