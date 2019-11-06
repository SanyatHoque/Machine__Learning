function [penalty_factor1, penalty_factor2] = penalty(a,b,f,pin)
    
% population(Individual_index).connection_matrix = a;
% population(Individual_index).input_matrix = b;
%    aa = size(population(Individual_index).connection_matrix);
%    bb = size(population(Individual_index).input_matrix);
aa = size(a);
bb = size(b);
   size_a = 0;
   for k = 1:aa(1)
       for kk = 1:aa(2)
           if (a(k,kk) ~= 0)
           size_a = size_a + 1;  
           end
       end
   end
   Neuron_Connections = size_a;
   Number_of_layers = 1;
   total_dim = (1 * (f + 1)) + (Number_of_layers * f * (f + 1));
   penalty_factor2 = 10*(((total_dim - Neuron_Connections)*pin)/(total_dim));
   size_b = 0;
%  for k = 1:bb(2)
   for kk = 1:bb(1)
       if (b(kk) ~= 0)
           size_b = size_b + 1;
       end
   end
%  end
   Input_Connections = size_b ;
%    Gen = Gen + 1;
   penalty_factor1 = 10*(((f-Input_Connections)*pin)/f);