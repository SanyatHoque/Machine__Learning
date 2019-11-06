%Main_code
% Benchmark NE with fixed topology, produces 1 indivual per Generation,
% Random Mutation, No Crossover
% Direct Encoding
% Genomes are passed from Gen i to Gen i+1
% With Dropout method for better Generalization 
close all;clear all;clc;
features = 2;  %Total Number of features in dataset
rows1 = 4;   %  Total Rows, % PLEASE NOTE, THIS CODE CAN WORK WITH OVER 10 MILLION ROWS
x1 = [0 0;0 1;1 0;1 1]';
% x1 = abs(randn(features,rows1));   %Simulated Dataset
% x = [ones(1,rows1);x1];
% y  = randi([0 1],1,rows1);
y = [1 0 0 1]';
Number_of_generations = 0;
iter = 5000;
Fitness_value = 1;
index=0;
j = 0;
while (Fitness_value < 100)  %&&  (j<=8)
    j = j + 1;
    switch j
        case 1
            jj= j
            Number_of_generations = Number_of_generations+1;
            [Accuracy,W1,W2] = mod5RBFawid7g(features,rows1,x1,y,iter);  %1 hidden Layer NN
            Hidden_layer = 1;
        case 2
            jj= j
            Number_of_generations = Number_of_generations+1;
            [Accuracy,W1,W1a,W2] = modRBFawid8g(features,rows1,x1,y,iter);  %1 hidden Layer NN
            Hidden_layer = 1;    
        case 3
            jj= j
            Number_of_generations = Number_of_generations+1;
            [Accuracy,W1,W1a,W1b,W2] = modRBFawid9d(features,rows1,x1,y,iter,W1,W1a,W2);  %2 hidden Layer MLP
            Hidden_layer = 2 ;
        case 4
            jj= j
            Number_of_generations = Number_of_generations+1;
            [Accuracy,W1,W1a,W1b,W1c,W2] =  modRBFawid9g(features,rows1,x1,y,iter,W1,W1a,W1b,W2);  %3 hidden Layer MLP
            Hidden_layer = 3 ;
        case 5
            jj= j
            [Accuracy,W1,W1a,W1b,W1c,W1d,W1e,W2] =  modRBFawid9k(features,rows1,x1,y,iter,W1,W1a,W1b,W1c,W2);  %5 hidden Layer MLP
            Hidden_layer = 5 ;
        case 6
            jj= j
            [Accuracy,W1,W1a,W1b,W1c,W1d,W1e,W1f,W1g,W2] =  modRBFawid9m(features,rows1,x1,y,iter,W1,W1a,W1b,W1c,W1d,W1e,W2);  %7 hidden Layer MLP
            Hidden_layer = 7 ;
        case 7
            while (Fitness_value < 95) %&& (Number_of_generations<10)
                jj= j
                Number_of_generations = Number_of_generations+1;
%                 Hidden_layer = 13 ;
                Hidden_layer = 7 ;
                [Accuracy,W1,W1a,W1b,W1c,W1d,W1e,W1f,W1g,W2] =  modRBFawid9m(features,rows1,x1,y,iter,W1,W1a,W1b,W1c,W1d,W1e,W2);  %7 hidden Layer MLP
%                 [Accuracy,W1,W1a,W1b,W1c,W1d,W1e,W1f,W1g,W1h,W1i,W1j,W1k,W1l,W2] =  modRBFawid9o(features,rows1,x1,y,iter,W1,W1a,W1b,W1c,W1d,W1e,W1f,W1g,W2);  %13 hidden Layer MLP
Fitness_and_Index_Individual4 = [Fitness_value,Hidden_layer,Number_of_generations]
            end
    end
    Fitness_value = Accuracy;  
    Number_of_Layers = Hidden_layer;
    index=index+1;
    Fitness_and_Index_Individual4 = [Fitness_value,Hidden_layer,Number_of_generations]
Fitness_and_Index_Individual3(Number_of_generations) = Fitness_and_Index_Individual4(1);
end
 Fitness_and_Index_Individual4 = [Fitness_value,Hidden_layer,Number_of_generations]
 toc
 plot(Fitness_and_Index_Individual3(1:Number_of_generations))