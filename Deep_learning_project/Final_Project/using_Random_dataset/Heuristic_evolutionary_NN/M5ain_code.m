%Main_code   % Latest_Heuristic_NE
%% 3 Individuals, Performance and corresponding Index Individual for each Gen, Connection & Input Connections are added
%  Present Code: Gen and Speciation is different. 
%  Speciation could be same, Gen might be different
%  Speciation depends on Connection matrix and input Array(In)
%%
close all;clear all;clc;
iter = 120;
% format short
features = 26;  %Total Number of features in dataset
rows1 = 5000;   %  Total Rows, % PLEASE NOTE, THIS CODE CAN WORK WITH OVER 10 MILLION ROWS
x1 = abs(randn(features,rows1));   %Simulated Dataset
% x1 = [0 0;0 1;1 0;1 1]';
% x = [x1;ones(1,rows1)];    
y  = randi([0 1],1,rows1);
% alpha = 1.2;
% y = [1 0 0 1]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all;clear all;clc;
% clear all
% load main_data27.mat
% load main_data3.mat

% % iter = 200;
% % features = 25;   %Number of features in dataset
% % rows1 = 300000;    %Number of rows in dataset
% % % % x1 = randi([0 10],f,rows1);
% % % % x1 = load('5layerMLP_dataset.txt');   % Simulated Dataset, normalized
% % x1 = main_data3(1:rows1,1:features)';
% % % x1 = abs(randn(f,rows1));
% % % x1 = [0 0;0 1;1 0;1 1]';
% % x = [x1;ones(1,rows1)];
% % y = main_data6(1:rows1)';
% % % y = randi([0 1],1,rows1); %zeros (1,rows1)%
% % % y = [1 0 0 1 ];
% % % alpha = 0.059; %Learning Rate
% % % alpha = 0.01;
% % % ind = 1 ;
% % Individual = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
generation = 0;
Fitness = 1;
index=0;
j = 0;
    population(1).species = 1;
%   population(1).speciation = 1;
    population(1).Accuracy = 1;
    population(1).connection_matrix = 1;
    population(1).input_matrix = 1;
    population(1).false_pos_rate = 1;
tic
% while  (j<=8)
while (Fitness < 90)  %&&  (j<=8)
    j = j + 1;
    if (j==1)
        j = 1;
    elseif (j==2)
        j = 2;
    elseif (j==3)
        j = 3;
    elseif (j==4)
        j = 4;
    elseif (j==5)
        j = 5;
    elseif (j==6)
        j = 6;
    else (j==7)
        j = 1;
    end
    switch j
        case 1
            flagjj= j
            generation = generation+1;
            [highest_accuracy,population,Individual_index] = mod5RBFawid7g(features,rows1,x1,y,iter);  %1 hidden Layer NN
%             population1;
            Hidden_layer = 1;
        case 2
            flagjj= j
            generation = generation+1;
            [highest_accuracy,population,Individual_index] = mod5RBFawid8g(features,rows1,x1,y,iter);  %2 hidden Layer NN
%             population2;
            Hidden_layer = 2;
        case 3
            flagjj= j
            generation = generation+1;
            [highest_accuracy,population,Individual_index] = mod5RBFawid9d(features,rows1,x1,y,iter);  %3 hidden Layer MLP
            Hidden_layer = 3 ;
        case 4
            flagjj= j
            generation = generation+1;
            [highest_accuracy,population,Individual_index] =  mod5RBFawid9g(features,rows1,x1,y,iter);  %4 hidden Layer MLP
            Hidden_layer = 4 ;
        case 5
            flagjj= j
            generation = generation+1;
            [highest_accuracy,population,Individual_index] =  mod5RBFawid9k(features,rows1,x1,y,iter);  %5 hidden Layer MLP
            Hidden_layer = 5 ;
        case 6
            flagjj= j
            generation = generation+1;
            [highest_accuracy,population,Individual_index] =  mod5RBFawid9m(features,rows1,x1,y,iter);  %7 hidden Layer MLP
%           [Final_Accuracy,W1,W1a,W1b,W1c,W1d,W1e,W2] =  mod3RBFawid9m(features,rows1,x1,y,iter);  %7 hidden Layer MLP
            Hidden_layer = 7 ;
%             population = population6;
%         case 7
%            %while (Fitness_value < 80) %&& (generation<10)
%             jj= j
%             generation = generation+1;
%             [Final_Accuracy] =  mod3RBFawid9m(features,rows1,x1,y,iter);  %7 hidden Layer MLP            
% %           [Final_Accuracy,W1,W1a,W1b,W1c,W1d,W1e,W2] =  mod3RBFawid9m(features,rows1,x1,y,iter,W1,W1a,W1b,W1c,W1d,W1e,W2);  %7 hidden Layer MLP
%             Hidden_layer = 7 ;
%             %info = [Fitness_value,generation]
%             %end
    end
    Fitness = highest_accuracy;  
    Fitness_and_Index_Individual = [Fitness,generation,Individual_index]
Fitness_and_Index_Individual2(generation) = Fitness_and_Index_Individual(1);
end
 Fitness_Index_Individual = [Fitness,generation,Individual_index]
 toc
 plot(Fitness_and_Index_Individual2(1:generation))