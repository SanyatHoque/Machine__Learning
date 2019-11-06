% function [highest_accuracy,Individual_index] =  mod5RBFawid9m(f,rows1,x1,y,iter);
%7 hidden Layer MLP, No dropout included
%
% clear all
% load main_data27.mat
% load main_data3.mat

close all;clear all;clc;
iter = 150;
f = 26;   %Number of features in dataset
rows1 = 5000;    %Number of rows in dataset
% % x1 = randi([0 10],f,rows1);
% % x1 = load('5layerMLP_dataset.txt');   % Simulated Dataset, normalized
% x1 = main_data3(1:rows1,1:f)';
x1 = abs(randn(f,rows1));
% x1 = [0 0;0 1;1 0;1 1]';
x = [x1;ones(1,rows1)];
% y = main_data6(1:rows1)';
y = randi([0 1],1,rows1); %zeros (1,rows1)%
% y = [1 0 0 1 ];
% alpha = 0.059; %Learning Rate
alpha = 0.9;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all;clear all;clc;
% clear all
% load main_data27.mat
% load main_data3.mat

% % % iter = 150;
% % % f = 5;   %Number of features in dataset
% % % rows1 = 30;    %Number of rows in dataset
% % % % % x1 = randi([0 10],f,rows1);
% % % % % x1 = load('5layerMLP_dataset.txt');   % Simulated Dataset, normalized
% % % x1 = main_data3(1:rows1,1:f)';
% % % % x1 = abs(randn(f,rows1));
% % % % x1 = [0 0;0 1;1 0;1 1]';
% % % x = [x1;ones(1,rows1)];
% % % y = main_data6(1:rows1)';
% % % % y = randi([0 1],1,rows1); %zeros (1,rows1)%
% % % % y = [1 0 0 1 ];
% % % % alpha = 0.059; %Learning Rate
% % % alpha = 2.43;
% % % % ind = 1 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Individual = 1;
Gen = 0;   % Initializing Generalization
    population(Individual).species = 1;   % Initializing Population
%   population(Individual).speciation = 1;
    population(Individual).Accuracy = 1;
    population(Individual).connection_matrix = 1;  % Initializing Chromosomes
    population(Individual).input_matrix = 1;
%     population(Individual).Total_penalty = 1;
%   Chromosomes  = 1;
    Individual_index = 1;
    penalty_factor1 = -5; penalty_factor2 = -5;
%% Random Values Weights,
W1 = randi([-1 1],f+1,f)';
W1a = randi([-1 1],f+1,f)'; 
W1b = randi([-1 1],f+1,f)';
W1c = randi([-1 1],f+1,f)';
W1d = randi([-1 1],f+1,f)';
W1e = randi([-1 1],f+1,f)';
W1f = randi([-1 1],f+1,f)';
W1g = randi([-1 1],f+1,f)';
% W1h = randi([-1 1],f+1,f)';
% W1i = randi([-1 1],f+1,f)';
% W1j = randi([-1 1],f+1,f)';
% W1k = randi([-1 1],f+1,f)';
% W1l = randi([-1 1],f+1,f)';
W2 =  randi([-1 1],f+1,1)';
population(Individual_index).connection_matrix = [W1' W1a' W1b' W1c' W1d' W1e' W1f' W1g' W2'];
% %% Start Speciation
% for Speciation = 1:2
%     
%% Start Individual
% flag2 = 0;
% Individual = 0;
% for Individual = 1:2
% flag2 = 1
tic
while (population(Individual).Accuracy < 90)
    % for ii = 1:8    
%% Chromosomes are passed from Previous Gen's Best Individual to current Gen
Individual_index; %Test  % Best Individual_index
population(Individual_index);  %Test % Best Individual_index 
% population(Individual_index).connection_matrix = Phenotype_Tree_matrix;
W1  = population(Individual_index).connection_matrix(1:f+1,(f*(1)-(f-1)):1*f)';
W1a = population(Individual_index).connection_matrix(1:f+1,(f*(2)-(f-1)):2*f)';
W1b = population(Individual_index).connection_matrix(1:f+1,(f*(3)-(f-1)):3*f)';
W1c = population(Individual_index).connection_matrix(1:f+1,(f*(4)-(f-1)):4*f)';
W1d = population(Individual_index).connection_matrix(1:f+1,(f*(5)-(f-1)):5*f)';
W1e = population(Individual_index).connection_matrix(1:f+1,(f*(6)-(f-1)):6*f)';
W1f = population(Individual_index).connection_matrix(1:f+1,(f*(7)-(f-1)):7*f)';
W1g = population(Individual_index).connection_matrix(1:f+1,(f*(8)-(f-1)):8*f)';
W2  = population(Individual_index).connection_matrix(1:f+1,(f*(9)-(f-1)):(9*f-(f-1)))';
for Individual = 1:5
%%
min_input_features = 4 ;  %min input feats
in2 = randi([0 1],1,f-min_input_features)';
in1 = randi([1 1],1,min_input_features)';
in = [in1;in2;1];
total_in = sum(in);
input_features = total_in;   %input feats for particular epoch
pin = 1;
% penalty_factor1 = 1 - ((input_features-min_input_features)*pin);
No_weights = 9-1;
connection_matrix = [randi([1 1],f+1,1),randi([0 1],f+1,f*No_weights)];
MG = 0.5;
%%
rows = 1;
% n = rows1/10;
n = rows1;
i = 1;
[Phenotype_Tree_matrix,Phenotype_Neuron_Connections,Root_Connection] = Tree_matrix4(f);  % Tree_matrix5 when usng XOR 
Weight_matrix = [W1',W1a',W1b',W1c',W1d',W1e',W1f',W1g',W2'].*Phenotype_Tree_matrix;
Accuracy(i) = 1;
while (i < iter)
% [Phenotype_Tree_matrix,Phenotype_Neuron_Connections,Root_Connection] = Tree_matrix3(f);
% Root = Root_Connection   %Test
% while (Accuracy(i) > 80) && (i < iter)
% flag2 = 2
% for i = 1:iter
    for j = 1:rows1/n
        m1 = n*(j-1)+1;
        m2 = n*(j-1)+n;       
        for m = m1:m2 
            %% Forward Propagation
            %% 1st Layer
%           x(:,m) = x(:,m).*in ; 
            a1 = W1*x(:,m);
            h1 = 1./(1+exp(-a1));
            h1 = [h1; ones(1,rows)];
            %% 2nd Layer
            a2 = W1a*h1;
            h2 = 1./(1+exp(-a2));
            h2 = [h2; ones(1,rows)];            
            %% 3rd Layer
            a3 = W1b*h2;
            h3 = 1./(1+exp(-a3));
            h3 = [h3; ones(1,rows)];            
            %% 4rth Layer
            a4 = W1c*h3;
            h4 = 1./(1+exp(-a4));
            h4 = [h4; ones(1,rows)];            
             %% 5th Layer
            a5 = W1d*h4;
            h5 = 1./(1+exp(-a5));
            h5 = [h5; ones(1,rows)];            
              %% 6th Layer
            a6 = W1e*h5;
            h6 = 1./(1+exp(-a6));
            h6 = [h6; ones(1,rows)];      
            %% 7th Layer
            a7 = W1f*h6;
            h7 = 1./(1+exp(-a7));
            h7 = [h7; ones(1,rows)];            
            %% 8th Layer
            a8 = W1g*h7;
            h8 = 1./(1+exp(-a8));
            h8 = [h8; ones(1,rows)];                       
            %% Output Layer
            %         a1_3 = W11_3*h1_2 + W12_3*h2_2 + b11_3; %Output Neuron 1
            %         h1_3(m) = 1/(1+exp(-a1_3)); %Output Neuron 1 Hypothesis
            %         del1_3 = (h1_3(m) - y(m))*h1_3(m)*(1-h1_3(m)); % del function
            aa1_3 = W2*h8;
            h11_3 = 1./(1+exp(-aa1_3));
            v1 = (1-h11_3)'*h11_3;
            v2 = (h11_3' - y(m));
            del11_3 = v1*v2;
            %% Backward Propagation
            % Output Layer Weights
            a22 = h8*del11_3;
            W2 = W2'-alpha.*a22;            
            %% 8th Hidden Layer Weights
            uu1 = h8(1:end-1,:)'*(1-h8(1:end-1,:));
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
            W1b = W1b' - alpha.*hh4;            
            %% 2nd Hidden Layer Weights
            ii1 = hh1*h2(1:end-1,:)'*(1-h2(1:end-1,:));
            ii2 = del11_3*W2(1:end-1)';
            ii3 = ii1*ii2;
            ii4 = x(:,m)*ii3;
            W1a = W1a' - alpha.*ii4;            
            %% 1st Hidden Layer Weights
            kk1 = ii1*h1(1:end-1,:)'*(1-h1(1:end-1,:)); 
            kk2 = del11_3*W2(1:end-1)';
            kk3 = kk1*kk2;
            kk4 = x(:,m)*kk3;
            W1 = W1' - alpha.*kk4;            
%%            %%
            W1 = W1';
            W1a = W1a';
            W1b = W1b';
            W1c = W1c';
            W1d = W1d';
            W1e = W1e';
            W1f = W1f';
            W1g = W1g';
%             W1h = W1h';
%             W1i = W1i';
%             W1j = W1j';
%             W1k = W1k';
%             W1l = W1l';
            W2 = W2';
            Weight_matrix = [W1',W1a',W1b',W1c',W1d',W1e',W1f',W1g',W2'];
%           Weight_matrix = [W1',W1a',W1b',W1c',W1d',W1e',W1f',W1g',W2'].*connection_matrix;
%             Output_Tree_matrix   % Test
%             Weight_matrix = [W1',W1a',W1b',W1c',W1d',W1e',W1f',W1g',W2'].*Phenotype_Tree_matrix;
            new_h11_3(m) = h11_3;
            new_y(m) = y(m);
            test1 = [new_h11_3 y(m)];
   end
end
   % test = [h11_3' y(m)]
    i = i + 1 ; 
    Hidden_8_Layer_Structure = [new_h11_3(:) new_y(:)]
    Final_Weight_matrix = Weight_matrix;
% end
%% End Main Algorithm. Start Assessment

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

Accuracy1 = ((true_pos_count+true_neg_count)/(true_pos_count+true_neg_count+false_neg_count+false_pos_count))*100;
a =  Final_Weight_matrix;
b =  in;
[penalty_factor1, penalty_factor2]=penalty(a,b,f,pin);
Accuracy(i) = Accuracy1 + penalty_factor1 + penalty_factor2;
false_pos_rate = false_pos_count/(false_pos_count+true_neg_count);
 if Accuracy(i) > 95;
        flag2 = 5;
         population(Individual).species = Individual;
%     population(Individual).speciation = 1;
    population(Individual).Accuracy = Accuracy(i);
    population(Individual).connection_matrix = Final_Weight_matrix;
    population(Individual).input_matrix = in;
    population(Individual).false_pos_rate = false_pos_rate;
           break 
    end 
%% Start Mutation or Crossover, Iteration i NOt Ended
flag = 0;
if Accuracy(i) <= Accuracy(i-1)
    MG = MG + 0.5;
    P = (MG+1)*0.05;
size_Weight_matrix = size(Weight_matrix);
Mutation_sd_matrix = P.*(randi([-1 1],size_Weight_matrix(1),size_Weight_matrix(2))) ;
Mutation_rate = 0.09;
Mutation_sd = Weight_matrix + Mutation_rate.*Mutation_sd_matrix ;
%Either Mutate, Node add or delete
%% Either Mutate, Node add or delete
 if (Individual > 3) && (Accuracy(i) <= Accuracy(i-1))   
    switch1 =  randi([1 2],1,1);
 else
    switch1 =  2;
 end
   switch switch1
%      switch 1
     case 1
%% Crossover
flag9 = 1;
size1 = round((size_Weight_matrix(1)-1)/2);
size2 = size_Weight_matrix(2);   % round((size_Weight_matrix(2)-1)/2);  %Width
Test1 = Final_Weight_matrix;  %Test %population(Individual).connection_matrix   %Test
Test2 = population(Individual-1).connection_matrix ;  %Test
dummy_matrix1 =  Final_Weight_matrix(1:size1,1:size2);
dummy_matrix2 =  population(Individual-1).connection_matrix(1:size1,1:size2);
Final_Weight_matrix = [dummy_matrix2;Test1(1:f+1-size1,1:size2)] ;
% population(Individual-1).connection_matrix = [dummy_matrix1;Test2(1:f+1-size1,1:size2)] ;
Weight_matrix = Final_Weight_matrix;
        case 2
%% Mutate
% if Accuracy(i) <= Accuracy(i-1)
    flag9 = 2;
    switch randi([1 3],1,1)
  % switch 2
        case 1
            Weight_matrix = Mutation_sd;   %Mutate
        case 2
            %%  Node addition
            for h = 1:size_Weight_matrix(1)
                for l = 1:size_Weight_matrix(2)
                    if (Weight_matrix(h,l) == 0)
                        Weight_matrix(h,l) = randi([-1 1],1,1);
                    end
                end
            end
        case 3
            %%  Node deletion
            for h = 1:size_Weight_matrix(1)
                for l = 1:size_Weight_matrix(2)
                    if (Weight_matrix(h,l) ~= 0)
                        Weight_matrix(h,l) = Weight_matrix(h,l).*randi([0 1],1,1);
                    end
                end
            end
    end
    end
elseif Accuracy(i) >  Accuracy(i-1)
    MG = 0;
    flag11 = 2;
    flag = 2;
end
%% End Mutation or Crossover
% Weight_matrix     %Test
% ind(i).Weight_matrix = Weight_matrix; 
end
% ind(i).Weight_matrix    %Test
% Individual = Individual + 1    %Test
% Accuracy(i)  %Test
%% end Each Individual 1
    population(Individual).species = Individual;
%     population(Individual).speciation = 1;
    population(Individual).Accuracy = Accuracy(i);
    population(Individual).connection_matrix = Weight_matrix;
    population(Individual).input_matrix = in;
    population(Individual).false_pos_rate = false_pos_rate;
    for iii = 1:Individual
        Population_array(iii) = [population(iii)] ;        
%         Population_array(iii) = [population(iii)] ;
%         connection_matrix_array = population(Individual).connection_matrix
    end
    for kkk = 1:Individual
        Accuracy_array(kkk) = Population_array(kkk).Accuracy ;  
        Individual__index_array(kkk) = Population_array(kkk).species ; 
%       Connection_array(kkk) =  Population_array(kkk).connection_matrix;
%       Input_array(kkk) =  Population_array(kkk).input_matrix;
    end
    
%     Final_Accuracy = max(Accuracy_array)
%     if population(Individual).Accuracy > 95;
%         flag2 = 5;
%         break 
%     end 
    %% Assessing Each Individual   %% end Each Individual 2
%     Accuracy_array;
%     Individual__index_array;
%   Connection_array
%   Input_array
%     Performance_array = [Accuracy_array;Individual__index_array];
%     Performance_array1 = Performance_array(1,1:Individual);
%     highest_accuracy = max(Performance_array1);
%     for l = 1:2
%     for ll = 1:Individual
%         if Performance_array(l,ll) == highest_accuracy;
%             Individual_index = ll ;  %Individual_index
%             %Accuracy_and_Index(l,ll) =
%             break
%         end
%     end
%     end
% for u = 1:Individual
   c =  population(Individual).connection_matrix;
   d =  population(Individual).input_matrix;
% end
   cc = size(population(Individual).connection_matrix);
   dd = size(population(Individual).input_matrix);
   size_c = 0;
   for k = 1:cc(1)
       for kk = 1:cc(2)
           if (c(k,kk) ~= 0)
           size_c = size_c + 1;  
           end
       end
   end
% for u = 1:Individual
%   Neuron_Connections_each(Individual) = size_c
   Neuron_Connections_each = size_c;
% end
   size_d = 0;
%  for k = 1:bb(2)
   for kk = 1:dd(1)
       if (d(kk) ~= 0)
           size_d = size_d + 1;
       end
   end
%  end
% for u = 1:Individual
%   Input_Connections_each(Individual) = size_d
    Input_Connections_each = size_d;
% end
%  population1 = population; 
Total_neurons(Individual) = Neuron_Connections_each + Input_Connections_each;
% Threshold = 20 ;
% if (Total_neurons(Individual) > Threshold);
%     population(Individual).speciation = 1;
%     else (Total_neurons(Individual) < Threshold);
%         population(Individual).speciation = 2;
% end
    %%
%     if (population(Individual).Accuracy > 95)
%     flag101 = 1 
%     break
%     end
end

%% end All Individual 
    Accuracy_array;
    Individual__index_array;
%   Connection_array
%   Input_array
    Performance_array = [Accuracy_array;Individual__index_array];
    Performance_array1 = Performance_array(1,1:Individual);
    highest_accuracy = max(Performance_array1);
    for l = 1:2
    for ll = 1:Individual
        if Performance_array(l,ll) == highest_accuracy;
            Individual_index = ll ;  %Individual_index
            %Accuracy_and_Index(l,ll) =
            break
        end
    end
    end
%    a =  population(Individual_index).connection_matrix;
%    b =  population(Individual_index).input_matrix;
%    aa = size(population(Individual_index).connection_matrix);
%    bb = size(population(Individual_index).input_matrix);
%    size_a = 0;
%    for k = 1:aa(1)
%        for kk = 1:aa(2)
%            if (a(k,kk) ~= 0)
%            size_a = size_a + 1;  
%            end
%        end
%    end
%    Neuron_Connections = size_a;
%    Number_of_weights = 8;
%    total_dim = (Number_of_weights) * ((Number_of_weights * f) + 1);
%    penalty_factor2 = 10*(((total_dim - Neuron_Connections)*pin)/(total_dim));
%    size_b = 0;
% %  for k = 1:bb(2)
%    for kk = 1:bb(1)
%        if (b(kk) ~= 0)
%            size_b = size_b + 1;
%        end
%    end
%  end
%    Input_Connections = size_b ;
   Gen = Gen + 1
%    penalty_factor1 = 10*(((f-Input_Connections)*pin)/f);
%  Chromosomes = [population(Individual_index).connection_matrix   population(Individual_index).input_matrix] ;   % Chromosomes are passed to Gen i+1
Performance_array3 = [highest_accuracy Gen Individual_index];
Performance_array4(Gen) = highest_accuracy
% if Gen>400
%     break
% end
end
toc
plot(Performance_array4(1:Gen))