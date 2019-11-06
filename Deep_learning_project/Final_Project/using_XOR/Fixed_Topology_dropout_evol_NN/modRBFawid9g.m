function [Accuracy,W1,W1a,W1b,W1c,W2] =  modRBFawid9g(f,rows1,x1,y,iter,W1,W1a,W1b,W2);
%3 hidden Layer MLP

% close all;clear all;clc;
% f = 26;   %Number of features in dataset
% rows1 = 30;    %Number of rows in dataset
% % x1 = randi([0 10],f,rows1);
% % x1 = load('5layerMLP_dataset.txt');   % Simulated Dataset, normalized
% x1 = abs(randn(f,rows1));
x = [x1;ones(1,rows1)] ;
% y  = [0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 1 0 1 1 0 1 0 0 0 0 1 0 0 1 0];
alpha = 0.9; %Learning Rate
%% Random Values Weights,
% % % % W1 = randi([-1 1],f+1,f)';
% % % % W1a = randi([-1 1],f+1,f)';
% % % % W1b = randi([-1 1],f+1,f)';
W1c = randi([-1 1],f+1,f)';
% W1d = randi([-1 1],f+1,f)';
% W1e = randi([-1 1],f+1,f)';
% W1f = randi([-1 1],f+1,f)';
% W1g = randi([-1 1],f+1,f)';
% W1h = randi([-1 1],f+1,f)';
% W1i = randi([-1 1],f+1,f)';
% W1j = randi([-1 1],f+1,f)';
% W1k = randi([-1 1],f+1,f)';
% W1l = randi([-1 1],f+1,f)';
% % % % % % % W2 =  randi([-1 1],f+1,1)';
rows = 1;
% n = rows1/10;
n = rows1;
for i = 1:iter
    for j = 1:rows1/n
        m1 = n*(j-1)+1;
        m2 = n*(j-1)+n;       
        for m = m1:m2 
            %% Forward Propagation 
            % 1st Layer
            a1 = W1*x(:,m);
            h1 = 1./(1+exp(-a1));
            h1 = [h1; ones(1,rows)];
            %% 2nd Layer
            a2 = W1a*h1;
            random_pr = randi([0 1],1,f)';
            a2 = random_pr.*a2;
            h2 = 1./(1+exp(-a2));
            h2 = [h2; ones(1,rows)];            
            %% 3rd Layer
            a3 = W1b*h2;
            random_pr = randi([0 1],1,f)';
            a3 = random_pr.*a3;
            h3 = 1./(1+exp(-a3));
            h3 = [h3; ones(1,rows)];            
            %% 4rth Layer
            a4 = W1c*h3;
            random_pr = randi([0 1],1,f)';
            a4 = random_pr.*a4;
            h4 = 1./(1+exp(-a4));
            h4 = [h4; ones(1,rows)];                      
            %% Output Layer
            aa1_3 = W2*h4;
            h11_3 = 1./(1+exp(-aa1_3));
            v1 = (1-h11_3)'*h11_3;
            v2 = (h11_3' - y(m));
            del11_3 = v1*v2;
            %% Backward Propagation
            % Output Layer Weights
            a22 = h4*del11_3;
            W2 = W2'-alpha.*a22;            
            %% 4rth Hidden Layer Weights
            uu1 = h4(1:end-1,:)'*(1-h4(1:end-1,:));
            uu2 = del11_3*W2(1:end-1)';
            uu3 = uu1*uu2;
            uu4 = x(:,m)*uu3;
            W1c = W1c' - alpha.*uu4;
            %% 3rd Hidden Layer Weights
            vv1 = uu1*h3(1:end-1,:)'*(1-h3(1:end-1,:));
            vv2 = del11_3*W2(1:end-1)';
            vv3 = vv1*vv2;
            vv4 = x(:,m)*vv3;
            W1b = W1b' - alpha.*vv4;
            %% 2nd Hidden Layer Weights
            qq1 = vv1*h2(1:end-1,:)'*(1-h2(1:end-1,:));
            qq2 = del11_3*W2(1:end-1)';
            qq3 = qq1*qq2;
            qq4 = x(:,m)*qq3;
            W1a = W1a' - alpha.*qq4;            
            %% 1st Hidden Layer Weights
            zz1 = qq1*h1(1:end-1,:)'*(1-h1(1:end-1,:));
            zz2 = del11_3*W2(1:end-1)';
            zz3 = zz1*zz2;
            zz4 = x(:,m)*zz3;
            W1 = W1' - alpha.*zz4;
%% %%       Mutation Start
            W1 = W1'.*rand(f,f+1);
            W1a = W1a'.*rand(f,f+1);
            W1b = W1b'.*rand(f,f+1);
            W1c = W1c'.*rand(f,f+1);
%             W1d = W1d';
%             W1e = W1e';
%             W1f = W1f';
%             W1g = W1g';
%             W1h = W1h';
%             W1i = W1i';
%             W1j = W1j';
%             W1k = W1k';
%             W1l = W1l';
            W2 = W2'.*rand(1,f+1);
            new_h11_3(m) = h11_3;
            new_y(m) = y(m);
            test1 = [new_h11_3 y(m)];
       %%   Mutation End
        end
    end
    % test = [h11_3' y(m)]
    Three_LayerANNtest = [new_h11_3(:) new_y(:)]
end
%%  TESTING 1/2 of all weights
            %% Forward Propagation 
            % 1st Layer
            a1 = 0.5.*W1*x(:,m);
            h1 = 1./(1+exp(-a1));
            h1 = [h1; ones(1,rows)];
            %% 2nd Layer
            a2 = 0.5.*W1a*h1;
%             random_pr = randi([0 1],1,26)';
%             a2 = random_pr.*a2;
            h2 = 1./(1+exp(-a2));
            h2 = [h2; ones(1,rows)];            
            %% 3rd Layer
            a3 = 0.5.*W1b*h2;
%             random_pr = randi([0 1],1,26)';
%             a3 = random_pr.*a3;
            h3 = 1./(1+exp(-a3));
            h3 = [h3; ones(1,rows)];            
            %% 4rth Layer
            a4 = 0.5.*W1c*h3;
%             random_pr = randi([0 1],1,26)';
%             a4 = random_pr.*a4;
            h4 = 1./(1+exp(-a4));
            h4 = [h4; ones(1,rows)];                      
            %% Output Layer
            aa1_3 = 0.5.*W2*h4;
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