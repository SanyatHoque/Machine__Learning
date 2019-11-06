function [highest_accuracy,population,Individual_index] =  mod5RBFawid7g(f,rows1,x1,y,iter);
%SINGLE SINGLE Layer Radial Basis Function

% close all;clear all;clc;
% iter = 5000
% f = 2;   %Number of features in dataset
% rows1 = 10;    %Number of rows in dataset
% x1 = randi([0 10],f,rows1);
% % % x1 = load('5layerMLP_dataset.txt');   % Simulated Dataset, normalized
% x1 = abs(randn(f,rows1));
x = [x1;ones(1,rows1)];
% y = randi([0 1],1,rows1);
alpha = 0.7; %Learning Rate
penalty_factor1 = -5; penalty_factor2 = -5;
% ind = 1 ;
% %% Start Speciation
% for Speciation = 1:2
%     
%% Start Individual
% flag2 = 0;
% Individual = 0;
% for Individual = 1:2
% flag2 = 1
for Individual = 1:3
%% Random Values Weights,
W1 = randi([-1 1],f+1,f)';
% W1a = randi([-1 1],f+1,f)';
W2 =  randi([-1 1],f+1,1)';
%%
min_input_features = f ;  %min input feats
in2 = randi([0 1],1,f-min_input_features)';
in1 = randi([1 1],1,min_input_features)';
in = [in1;in2;1];
total_in = sum(in);
input_features = total_in;   %input feats for particular epoch
pin = 1;
% penalty_factor1 = 1 - ((input_features-min_input_features)*pin);
No_weights = 2-1;
connection_matrix = [randi([1 1],f+1,1),randi([0 1],f+1,f*No_weights)];
MG = 0.01;
%%
rows = 1;
% n = rows1/10;
n = rows1;
i = 1;
Accuracy(i) = 1;
while (i < iter)
% while (Accuracy(i) > 80) && (i < iter)
% flag2 = 2
% for i = 1:iter
    for j = 1:rows1/n
        m1 = n*(j-1)+1;
        m2 = n*(j-1)+n;       
        for m = m1:m2 
            %% Forward Propagation
            %% 1st Layer
            x(:,m) = x(:,m).*in;
            a1 = W1*x(:,m);
            h1 = 1./(1+exp(-a1));
            h1 = [h1; ones(1,rows)];
             %% 2nd Layer
%             a2 = W1a*h1;
%             h2 = 1./(1+exp(-a2));
%             h2 = [h2; ones(1,rows)];
            %% Output Layer
            aa1_3 = W2*h1;
            h11_3 = 1./(1+exp(-aa1_3));
            v1 = (1-h11_3)'*h11_3;
            v2 = (h11_3' - y(m));
            del11_3 = v1*v2;
            %% Backward Propagation
            % Output Layer Weights
            a22 = h1*del11_3;
            W2 = W2'-alpha.*a22;
             %% 2nd Hidden Layer Weights
%             u1 = h2(1:end-1,:)'*(1-h2(1:end-1,:));
%             u2 = del11_3*W2(1:end-1)';
%             u3 = u1*u2;
%             u4 = x(:,m)*u3;
% %             u4 = [u4; ones(extra,f)];
%             W1a = W1a' - alpha.*u4;
            %% 1st Hidden Layer Weights
            s1 = h1(1:end-1,:)'*(1-h1(1:end-1,:))*h1(1:end-1,:)'*(1-h1(1:end-1,:));
            s2 = del11_3*W2(1:end-1)';
            s3 = s1*s2;
            s4 = x(:,m)*s3;
            W1 = W1' - alpha.*s4;
            %%
            Weight_matrix = [W1,W2];
            Weight_matrix = [W1,W2].*connection_matrix;
            W1 = W1';
%             W1a = W1a';
            W2 = W2';
            new_h11_3(m) = h11_3;
            new_y(m) = y(m);
            % test = [h11_3' y(m)]
        end
    end
   % test = [h11_3' y(m)]
    i = i + 1 ; 
    Single_Layer_test = [new_h11_3(:) new_y(:)]
    Final_Weight_matrix = Weight_matrix;
% end
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
false_pos_count = 0 ;
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
    
%     plot(Accuracy(1:i))
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
 if (Individual > 1) && (Accuracy(i) <= Accuracy(i-1))   
    switch1 =  randi([1 2],1,1);
 else
    switch1 =  2;
 end
     switch switch1
%    switch 1
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
population(Individual-1).connection_matrix = [dummy_matrix1;Test2(1:f+1-size1,1:size2)] ;
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
elseif Accuracy(i) > Accuracy(i-1)
    MG = 0;
    flag = 2;
end
%% End Mutation or Crossover
% Weight_matrix     %Test
% ind(i).Weight_matrix = Weight_matrix; 
end
% ind(i).Weight_matrix    %Test
% Individual = Individual + 1    %Test
% Accuracy(i)  %Test
%% end Each Individual 
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
    %% Assessing Each Individual
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
   for k = 1:cc(2)
       for kk = 1:cc(1)
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
% Threshold = mean(Total_neurons) ;
% if (Total_neurons(Individual) > Threshold);
%     population(Individual).speciation = 1;
%     else (Total_neurons(Individual) < Threshold);
%         population(Individual).speciation = 2;
% end
    %%
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
  