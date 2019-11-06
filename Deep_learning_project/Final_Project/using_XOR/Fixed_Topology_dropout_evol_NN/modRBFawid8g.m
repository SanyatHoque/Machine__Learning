function [Accuracy,W1,W1a,W2] =  modRBFawid8g(f,rows1,x1,y,iter);
%Single Layer Radial Basis Function

% close all;clear all;clc;
% iter = 10;
% f = 2;   %Number of features in dataset
% rows1 = 30;    %Number of rows in dataset
% % x1 = randi([0 10],f,rows1);
% % x1 = load('5layerMLP_dataset.txt');   % Simulated Dataset, normalized
% x1 = abs(randn(f,rows1));
x = [x1;ones(1,rows1)] ;
% y  = randi([0 1],1,rows1);
alpha = 0.9; %Learning Rate
%% Random Values Weights,
W1 = randi([-1 1],f+1,f)';
W1a = randi([-1 1],f+1,f)';
W2 =  randi([-1 1],f+1,1)';
rows = 1;
% n = rows1/10;
n = rows1;
for i = 1:iter
    for j = 1:rows1/n
        m1 = n*(j-1)+1;
        m2 = n*(j-1)+n;       
        for m = m1:m2 
            %% Forward Propagation
            %% 1st Layer
            a1 = W1*x(:,m);
            h1 = 1./(1+exp(-a1));
            h1 = [h1; ones(1,rows)];
            %% 2nd Layer
            a2 = W1a*h1;
            random_pr = randi([0 1],1,f)';
            a2 = random_pr.*a2;
            h2 = 1./(1+exp(-a2));
            h2 = [h2; ones(1,rows)];
            %% Output Layer
            aa1_3 = W2*h2;
            h11_3 = 1./(1+exp(-aa1_3));
            v1 = (1-h11_3)'*h11_3;
            v2 = (h11_3' - y(m));
            del11_3 = v1*v2;
            %% Backward Propagation
            % Output Layer Weights
            a22 = h2*del11_3;
            W2 = W2'-alpha.*a22;
            %% 2nd Hidden Layer Weights
            u1 = h2(1:end-1,:)'*(1-h2(1:end-1,:));
            u2 = del11_3*W2(1:end-1)';
            u3 = u1*u2;
            u4 = x(:,m)*u3;
%             u4 = [u4; ones(extra,f)];
            W1a = W1a' - alpha.*u4;
            %% 1st Hidden Layer Weights
            s1 = h2(1:end-1,:)'*(1-h2(1:end-1,:))*h1(1:end-1,:)'*(1-h1(1:end-1,:));
            s2 = del11_3*W2(1:end-1)';
            s3 = s1*s2;
            s4 = x(:,m)*s3;
            W1 = W1' - alpha.*s4;
            %%   Mutation Start
            W1 = W1'.*rand(f,f+1);
            W1a = W1a'.*rand(f,f+1);
            W2 = W2'.*rand(1,f+1);
            new_h11_3(m) = h11_3;
            new_y(m) = y(m);
            % test = [h11_3' y(m)]
            %%   Mutation End
        end
    end
   % test = [h11_3' y(m)]
    Two_LayerANNtest = [new_h11_3(:) new_y(:)]
end
%%  TESTING 1/2 of all weights
            %% Forward Propagation
            %% 1st Layer
            a1 = 0.5.*W1*x(:,m);
            h1 = 1./(1+exp(-a1));
            h1 = [h1; ones(1,rows)];
            %% 2nd Layer
            a2 = 0.5.*W1a*h1;
            random_pr = randi([0 1],1,f)';
            a2 = random_pr.*a2;
            h2 = 1./(1+exp(-a2));
            h2 = [h2; ones(1,rows)];
            %% Output Layer
            aa1_3 = 0.5.*W2*h2;
            h11_3 = 1./(1+exp(-aa1_3));
%% END
for mm = 1:rows1
    if (new_h11_3(mm)>0.5);
        new_new_h11_3(mm) = 1 ;
    else (new_h11_3(mm)<0.5);
        new_new_h11_3(mm) = 0 ;
    end
end
test2 = [new_new_h11_3' new_y'];
true_pos_count = 0 ;
true_neg_count = 0 ;
false_neg_count = 0 ;
false_pos_count = 0;
for mm = 1:rows1
    if (new_new_h11_3(mm)==1 && y(mm)==1);
        true_pos_count = true_pos_count + 1;
    end
    if (new_new_h11_3(mm)==0 && y(mm)==0);
        true_neg_count = true_neg_count + 1;
    end
    if (new_new_h11_3(mm)==1 && y(mm)==0);
        false_pos_count = false_pos_count + 1;
    end
    if (new_new_h11_3(mm)==0 && y(mm)==1);
        false_neg_count = false_neg_count + 1;
    end
end
true_pos_count = true_pos_count;
true_neg_count = true_neg_count;
false_neg_count = false_neg_count;
false_pos_count = false_pos_count;

Accuracy = ((true_pos_count+true_neg_count)/(true_pos_count+true_neg_count+false_neg_count+false_pos_count))*100;