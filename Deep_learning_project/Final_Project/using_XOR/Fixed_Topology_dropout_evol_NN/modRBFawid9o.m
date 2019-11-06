function [Accuracy,W1,W1a,W1b,W1c,W1d,W1e,W1f,W1g,W1h,W1i,W1j,W1k,W1l,W2] =  modRBFawid9o(f,rows1,x1,y,iter,W1,W1a,W1b,W1c,W1d,W1e,W1f,W1g,W2);
%13 hidden Layer MLP

% close all;clear all;clc;
% f = 26;   %Number of features in dataset
% rows1 = 30;    %Number of rows in dataset
% % x1 = randi([0 10],f,rows1);
% % x1 = load('5layerMLP_dataset.txt');   % Simulated Dataset, normalized
% x1 = abs(randn(f,rows1));
x = [x1;ones(1,rows1)] ;
% y  = [0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 1 0 1 1 0 1 0 0 0 0 1 0 0 1 0];
alpha = 0.9;%alpha = 0.0498; %Learning Rate
%% Random Values Weights,
% % % % % % W1 = randi([-1 1],f+1,f)';
% % % % % % W1a = randi([-1 1],f+1,f)';
% % % % % % W1b = randi([-1 1],f+1,f)';
% % % % % % W1c =  randi([-1 1],f+1,f)';  % load('W1c_7layerMLP.txt')'; %
% % % % % % W1d = randi([-1 1],f+1,f)';   %load('W1d_7layerMLP.txt')'; 
% % % % % % W1e = randi([-1 1],f+1,f)';   %load('W1e_7layerMLP.txt')'; 
% % % % % % W1f = randi([-1 1],f+1,f)';    %load('W1f_7layerMLP.txt')'; %
% % % % % % W1g = randi([-1 1],f+1,f)';    %load('W1g_7layerMLP.txt')'; %
W1h = randi([-1 1],f+1,f)';
W1i = randi([-1 1],f+1,f)';
W1j = randi([-1 1],f+1,f)';
W1k = randi([-1 1],f+1,f)';
W1l = randi([-1 1],f+1,f)';
% % % % % % W2 =  randi([-1 1],f+1,1)';   %load('W2_7layerMLP.txt')';
% n = rows1/10;
n = rows1 ; 
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
             %% 5th Layer
            a5 = W1d*h4;
            random_pr = randi([0 1],1,f)';
            a5 = random_pr.*a5;
            h5 = 1./(1+exp(-a5));
            h5 = [h5; ones(1,rows)];            
              %% 6th Layer
            a6 = W1e*h5;
            random_pr = randi([0 1],1,f)';
            a6 = random_pr.*a6;
            h6 = 1./(1+exp(-a6));
            h6 = [h6; ones(1,rows)];      
            %% 7th Layer
            a7 = W1f*h6;
            random_pr = randi([0 1],1,f)';
            a7 = random_pr.*a7;
            h7 = 1./(1+exp(-a7));
            h7 = [h7; ones(1,rows)];            
            %% 8th Layer
            a8 = W1g*h7;
            random_pr = randi([0 1],1,f)';
            a8 = random_pr.*a8;
            h8 = 1./(1+exp(-a8));
            h8 = [h8; ones(1,rows)];    
             %% 8th Layer
            a9 = W1h*h8;
            random_pr = randi([0 1],1,f)';
            a9 = random_pr.*a9;
            h9 = 1./(1+exp(-a9));
            h9 = [h9; ones(1,rows)];    
             %% 8th Layer
            a10 = W1i*h9;
            random_pr = randi([0 1],1,f)';
            a10 = random_pr.*a10;
            h10 = 1./(1+exp(-a10));
            h10 = [h10; ones(1,rows)];    
             %% 8th Layer
            a11 = W1j*h10;
            random_pr = randi([0 1],1,f)';
            a11 = random_pr.*a11;
            h11 = 1./(1+exp(-a11));
            h11 = [h11; ones(1,rows)];    
             %% 8th Layer
            a12 = W1k*h11;
            random_pr = randi([0 1],1,f)';
            a12 = random_pr.*a12;
            h12 = 1./(1+exp(-a12));
            h12 = [h12; ones(1,rows)];    
%              %% 8th Layer
%             a13 = W1l*h12;
%             h13 = 1./(1+exp(-a13));
%             h13 = [h13; ones(1,rows)];    
            %% Output Layer
            %         a1_3 = W11_3*h1_2 + W12_3*h2_2 + b11_3; %Output Neuron 1
            %         h1_3(m) = 1/(1+exp(-a1_3)); %Output Neuron 1 Hypothesis
            %         del1_3 = (h1_3(m) - y(m))*h1_3(m)*(1-h1_3(m)); % del function
            aa1_3 = W2*h12;
            h11_3 = 1./(1+exp(-aa1_3));
            v1 = (1-h11_3)'*h11_3;
            v2 = (h11_3' - y(m));
            del11_3 = v1*v2;
            %% Backward Propagation
            % Output Layer Weights
            a22 = h12*del11_3;
            W2 = W2'-alpha.*a22;      
              %% 8th Hidden Layer Weights
%             dd1 = h12(1:end-1,:)'*(1-h12(1:end-1,:));
%             dd2 = del11_3*W2(1:end-1)';
%             dd3 = dd1*dd2;
%             dd4 = x(:,m)*dd3;
%             W1l = W1l' - alpha.*dd4;
%               %% 8th Hidden Layer Weights
%             sss1 = h12(1:end-1,:)'*(1-h12(1:end-1,:));
%             sss2 = del11_3*W2(1:end-1)';
%             sss3 = sss1*sss2;
%             sss4 = x(:,m)*sss3;
%             W1l = W1l' - alpha.*sss4;
             %% 8th Hidden Layer Weights
            ss1 = h12(1:end-1,:)'*(1-h12(1:end-1,:));
            ss2 = del11_3*W2(1:end-1)';
            ss3 = ss1*ss2;
            ss4 = x(:,m)*ss3;
            W1k = W1k' - alpha.*ss4;
             %% 8th Hidden Layer Weights
            tt1 = ss1*h11(1:end-1,:)'*(1-h11(1:end-1,:));
            tt2 = del11_3*W2(1:end-1)';
            tt3 = tt1*tt2;
            tt4 = x(:,m)*tt3;
            W1j = W1j' - alpha.*tt4;
             %% 8th Hidden Layer Weights
            ww1 = tt1*h10(1:end-1,:)'*(1-h10(1:end-1,:));
            ww2 = del11_3*W2(1:end-1)';
            ww3 = ww1*ww2;
            ww4 = x(:,m)*ww3;
            W1i = W1i' - alpha.*ww4;
             %% 8th Hidden Layer Weights
            ee1 = ww1*h9(1:end-1,:)'*(1-h9(1:end-1,:));
            ee2 = del11_3*W2(1:end-1)';
            ee3 = ee1*ee2;
            ee4 = x(:,m)*ee3;
            W1h = W1h' - alpha.*ee4;
            %% 8th Hidden Layer Weights
            uu1 = ee1*h8(1:end-1,:)'*(1-h8(1:end-1,:));
            uu2 = del11_3*W2(1:end-1)';
            uu3 = uu1*uu2;
            uu4 = x(:,m)*uu3;
            W1g = W1g' - alpha.*uu4;
            %% 7th Hidden Layer Weights
            vv1 = uu1*h7(1:end-1,:)'*(1-h7(1:end-1,:));
            vv2 = del11_3*W2(1:end-1)';
            vv3 = vv1*vv2;
            vv4 = x(:,m)*vv3;
            W1f = W1f' - alpha.*vv4;
            %% 6th Hidden Layer Weights
            qq1 = vv1*h6(1:end-1,:)'*(1-h6(1:end-1,:));
            qq2 = del11_3*W2(1:end-1)';
            qq3 = qq1*qq2;
            qq4 = x(:,m)*qq3;
            W1e = W1e' - alpha.*qq4;            
            %% 5th Hidden Layer Weights
            zz1 = qq1*h5(1:end-1,:)'*(1-h5(1:end-1,:));
            zz2 = del11_3*W2(1:end-1)';
            zz3 = zz1*zz2;
            zz4 = x(:,m)*zz3;
            W1d = W1d' - alpha.*zz4;
            %% 4th Hidden Layer Weights
            gg1 = zz1*h4(1:end-1,:)'*(1-h4(1:end-1,:));
            gg2 = del11_3*W2(1:end-1)';
            gg3 = gg1*gg2;
            gg4 = x(:,m)*gg3;
            W1c = W1c' - alpha.*gg4;
            %% 3rd Hidden Layer Weights
            hh1 =  gg1*h3(1:end-1,:)'*(1-h3(1:end-1,:));
            hh2 = del11_3*W2(1:end-1)';
            hh3 = hh1*hh2;
            hh4 = x(:,m)*hh3;
%             hh4 = [hh4; ones(1,f)];
            W1b = W1b' - alpha.*hh4;            
            %% 2nd Hidden Layer Weights
            ii1 = hh1*h2(1:end-1,:)'*(1-h2(1:end-1,:));
            ii2 = del11_3*W2(1:end-1)';
            ii3 = ii1*ii2;
            ii4 = x(:,m)*ii3;
%             ii4 = [ii4; ones(1,f)];
            W1a = W1a' - alpha.*ii4;            
            %% 1st Hidden Layer Weights
            kk1 = ii1*h1(1:end-1,:)'*(1-h1(1:end-1,:)); 
            kk2 = del11_3*W2(1:end-1)';
            kk3 = kk1*kk2;
            kk4 = x(:,m)*kk3;
            W1 = W1' - alpha.*kk4;            
%%   %%%%   Mutation Start
            W1 = W1'.*rand(f,f+1);
            W1a = W1a'.*rand(f,f+1);
            W1b = W1b'.*rand(f,f+1);
            W1c = W1c'.*rand(f,f+1);
            W1d = W1d'.*rand(f,f+1);
            W1e = W1e'.*rand(f,f+1);
            W1f = W1f'.*rand(f,f+1);
            W1g = W1g'.*rand(f,f+1);
            W1h = W1h'.*rand(f,f+1);
            W1i = W1i'.*rand(f,f+1);
            W1j = W1j'.*rand(f,f+1);
            W1k = W1k'.*rand(f,f+1);
%             W1l = W1l';
            W2 = W2'.*rand(1,f+1);
            new_h11_3(m) = h11_3;
            new_y(m) = y(m);
%             test1 = [new_h11_3 y(m)];
       %%   Mutation End
        end
    end
    % test = [h11_3' y(m)]
    Thirteen_LayerANNtest = [new_h11_3(:) new_y(:)]
end
%%  TESTING 1/2 of all weights
            %% 1st Layer
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
             %% 5th Layer
            a5 = 0.5.*W1d*h4;
%             random_pr = randi([0 1],1,26)';
%             a5 = random_pr.*a5;
            h5 = 1./(1+exp(-a5));
            h5 = [h5; ones(1,rows)];            
              %% 6th Layer
            a6 = 0.5.*W1e*h5;
%             random_pr = randi([0 1],1,26)';
%             a6 = random_pr.*a6;
            h6 = 1./(1+exp(-a6));
            h6 = [h6; ones(1,rows)];      
            %% 7th Layer
            a7 = 0.5.*W1f*h6;
%             random_pr = randi([0 1],1,26)';
%             a7 = random_pr.*a7;
            h7 = 1./(1+exp(-a7));
            h7 = [h7; ones(1,rows)];            
            %% 8th Layer
            a8 = 0.5.*W1g*h7;
%             random_pr = randi([0 1],1,26)';
%             a8 = random_pr.*a8;
            h8 = 1./(1+exp(-a8));
            h8 = [h8; ones(1,rows)];    
             %% 8th Layer
            a9 = 0.5.*W1h*h8;
%             random_pr = randi([0 1],1,26)';
%             a9 = random_pr.*a9;
            h9 = 1./(1+exp(-a9));
            h9 = [h9; ones(1,rows)];    
             %% 8th Layer
            a10 = 0.5.*W1i*h9;
%             random_pr = randi([0 1],1,26)';
%             a10 = random_pr.*a10;
            h10 = 1./(1+exp(-a10));
            h10 = [h10; ones(1,rows)];    
             %% 8th Layer
            a11 = 0.5.*W1j*h10;
%             random_pr = randi([0 1],1,26)';
%             a11 = random_pr.*a11;
            h11 = 1./(1+exp(-a11));
            h11 = [h11; ones(1,rows)];    
             %% 8th Layer
            a12 = 0.5.*W1k*h11;
%             random_pr = randi([0 1],1,26)';
%             a12 = random_pr.*a12;
            h12 = 1./(1+exp(-a12));
            h12 = [h12; ones(1,rows)];    
%              %% 8th Layer
%             a13 = W1l*h12;
%             h13 = 1./(1+exp(-a13));
%             h13 = [h13; ones(1,rows)];    
            %% Output Layer
            %         a1_3 = W11_3*h1_2 + W12_3*h2_2 + b11_3; %Output Neuron 1
            %         h1_3(m) = 1/(1+exp(-a1_3)); %Output Neuron 1 Hypothesis
            %         del1_3 = (h1_3(m) - y(m))*h1_3(m)*(1-h1_3(m)); % del function
            aa1_3 = 0.5.*W2*h12;
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