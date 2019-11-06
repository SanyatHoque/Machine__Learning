clear all;clc;
input_features = 26;   %input feats for particular epoch
min_input_features = 5 ;  %min input feats
pin = 0.80;
penalty_factor1 = 1 - ((input_features-min_input_features)*pin);

chide = 150;  %total hidden nodes
mhide =  50; %min hidden nodes
phide = 0.4;
penalty_factor2 = 1 - ((chide - mhide)*phide);

in2 = randi([0 1],1,26-5)';
in1 = randi([1 1],1,5)';
in = [in1;in2];
total_in = sum(in);

a = [1 2 3; 4 5 6; 7 8 9];
b = [10 11 12; 13 14 15; 16 17 18];
answer = a.*b;

for n = 1:3
% n = 1;
    population(n).species = 1:3
    population(n).Fitness = 1
    population(n).Accuracy = 80
    population(n).connection_matrix = 1:10
    population(n).input_matrix = 1:5
end
for n = 4:7
% n = 1;
    population(n).species = 4:7
    population(n).Fitness = 5
    population(n).Accuracy = 90
    population(n).connection_matrix = 1:10
    population(n).input_matrix = 1:5
end
for mm = 1:7
    if population(mm).Fitness < 3
        speciation(1).population = population(mm);
    else population(mm).Fitness > 3
        speciation(2).population = population(mm);
    end
end
% for m = 1
%     speciation(m).population = population(1:3);
% end
% for m = 2
%     speciation(m).population = population(4:7);
% end
% flag = 0;
% for mm = 1:7
%     if population(mm).Fitness > 1
%         flag = 1;
%     end
% end
% speciation(2).population ;
% for n = 1:3
%     population(n).species = 1:10
%     population(n).Fitness = 1
%     population(n).Accuracy = 2
%     population(n).connection_matrix = 1:10
%     population(n).input_matrix = 1:10
% end
% for n = 1:3
% % n = 1;
%     population(n).species = 1:3
%     population(n).Fitness = 1
%     population(n).Accuracy = 80
%     population(n).connection_matrix = 1:10
%     population(n).input_matrix = 1:5
% end
% for n = 4:7
% % n = 1;
%     population(n).species = 4:7
%     population(n).Fitness = 2
%     population(n).Accuracy = 90
%     population(n).connection_matrix = 1:10
%     population(n).input_matrix = 1:5
% end
% for mm = 1:7
%     if population(mm).Fitness < 1
%         speciation = 1;
%     else population(mm).Fitness > 1
%         speciation = 2;
%     end
% end