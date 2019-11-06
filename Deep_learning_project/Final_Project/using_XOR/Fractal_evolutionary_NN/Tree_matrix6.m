function [Final_ouput_matrix102,Neuron_Connections_tree_matrix,m] = Tree_matrix6(f);
% clear all;clc;
% f = 2;  
% W1 = randi([1 2],f+1,f);
New_W1 = ones(f+1,f);
% W1a = randi([1 2],f+1,f);
% % % % % New_W1a = zeros(f+1,f);
% W1b = randi([1 2],f+1,f);
% % % % New_W1b = zeros(f+1,f);
% W1c = randi([1 2],f+1,f);
% % % % New_W1c = zeros(f+1,f);
% % % % % W1d = randi([1 2],f+1,f);
% % % % New_W1d = zeros(f+1,f);
% % % % % W1e = randi([1 2],f+1,f);
% % % % New_W1e = zeros(f+1,f);
% % % % % W1f = randi([1 2],f+1,f);
% % % % New_W1f = zeros(f+1,f);
% % % % % W1g = randi([1 2],f+1,f);
% % % % New_W1g = zeros(f+1,f);
% W2 =  randi([1 2],f+1,1);
New_W2 = zeros(f+1,1);
% a = [ W11 W21
%       W12 W22 ];
% b = [ W11a W21a
%       W12a W22a ];
% c = [ W11b W21b
%       W12b W22b ];
% d = [ W2a
%       W2b ];
 % output_matrix = [randi([1 1],2,1) randi([0 1],2,1)];
 %% Root of Tree
% m = randi([1 (f-1)],1,1)         
% d(m) = true  
% ouput_column_element = d 
% %% 
% input_matrix = [randi([0 1],f,f)];  %Input Matrix
% %2nd Last Matrix, Create many branches
% dummy_column = b(1:end,m);   %m  
% b(1:end,m) = randi([0 1],2,1);
% %Middle Matrix
% dummy_diag = diag(c);
% dummy_element = dummy_diag(m); 
% % ouput_column_element =  d(m)
% % dummy_output_column = d(m);
% Final_ouput_matrix101 = [a b c d];
%%%%%%%%%%%%%%%%
Final_ouput_matrix101 = [New_W1 New_W2];
%% Root of Tree   Starting Process Output Layer
n0 = randi([2 2],1,1)   ;
m = randi([1 f-1],1,1)   ;
if (n0 == 2)
New_W2(m) = true  ;
New_W2(m+1) = true  ;
elseif (n0 == 3)   
New_W2(m) = true  ;
New_W2(m+1) = true  ;
New_W2(m+2) = true  ;
else (n0 == 4)   
New_W2(m) = true  ;
New_W2(m+1) = true  ;
New_W2(m+2) = true  ;
New_W2(m+3) = true  ;
end
New_W2;
m101 = sum(New_W2);
% % % % %% 8th Layer Second Last Matrix, Input Matrix nn = f column feature 
% % % % % New_W1 = [randi([0 1],f+1,f)]  %Input Matrix, Random Branches
% % % % 
% % % % for nn=1:m101    %Last Matrix, nn = f column feature
% % % % ss = New_W1g(:,nn) ;
% % % % % Synapses 2-4 from Nodes. For each column m101, it makes
% % % % % n1 number of branches
% % % % n1 = randi([2 2],1,1) ; 
% % % % for k = 1:n1 
% % % %     ss(k) = true;
% % % % %     ss %Test
% % % % end
% % % % New_W1g(:,nn) = ss; 
% % % % end
% % % % New_W1g;
% % % % a1 = 0;a2 = zeros(1,f);
% % % % for ii = 1:f
% % % % for jj = f+1
% % % % a1 = New_W1g(1:jj,ii);
% % % % end
% % % % a2(ii) = sum(a1);
% % % % end
% % % % m102 = max(a2);
% % % % %% 7th Layer middle Matrix, Make many Random branches, 7th Layer
% % % % % m102 = sum(New_W1(1,1:end));
% % % % for nn=1:m102    %2nd Last Matrix, nn = f column feature
% % % % ss = New_W1f(:,nn) ;
% % % % % Synapses 2-4 from Nodes. For each column m102, it makes
% % % % % n2 number of branches
% % % % n2 = randi([2 2],1,1) ; % Synapses 2-4 from Nodes
% % % % for k = 1:n2 
% % % %     ss(k) = true ;
% % % % %     ss %Test
% % % % end
% % % % New_W1f(:,nn) = ss ;
% % % % end
% % % % New_W1f;
% % % % a1 = 0;a2 = zeros(1,f);
% % % % for ii = 1:f
% % % % for jj = f+1
% % % % a1 = New_W1f(1:jj,ii);
% % % % end
% % % % a2(ii) = sum(a1);
% % % % end
% % % % m103 = max(a2);
% % % % % New_W1a    % Test
% % % % % %% 2nd Last Matrix, Same SUBSTRUCTURE Make 2 branches, 7th Layer
% % % % % for i = m
% % % % %     for j = m
% % % % % New_W1a(i,j) = true;
% % % % %     end
% % % % % end
% % % % % for i = m+1
% % % % %     for j = m+1
% % % % % New_W1a(i,j) = true;
% % % % %     end
% % % % % end
% % % % %% 6th LayerMiddle Matrix  1st, Same SUBSTRUCTURE Make 2 branches, 6th Layer
% % % % % m103 = sum(New_W1a(1,1:end));
% % % % for nn=1:m103    %Middle Matrix, nn = f column feature
% % % % ss = New_W1e(:,nn) ;
% % % % % Synapses 2-4 from Nodes. For each column m103, it makes
% % % % % n3 number of branches
% % % % n3 = randi([2 2],1,1);  % Synapses 2-4 from Nodes
% % % % for k = 1:n3 
% % % %     ss(k) = true ;
% % % % %     ss %Test
% % % % end
% % % % New_W1e(:,nn) = ss ;
% % % % end
% % % % New_W1e;
% % % % a1 = 0;a2 = zeros(1,f);
% % % % for ii = 1:f
% % % % for jj = f+1
% % % % a1 = New_W1e(1:jj,ii);
% % % % end
% % % % a2(ii) = sum(a1);
% % % % end
% % % % m104 = max(a2);
% % % % %% 5th Layer%Middle Matrix  1st, Same SUBSTRUCTURE Make 2 branches, 5th Layer
% % % % % m104 = sum(New_W1b(1,1:end));
% % % % for nn=1:m104    %Middle Matrix, nn = f column feature
% % % % ss = New_W1d(:,nn) ;
% % % % % Synapses 2-4 from Nodes. For each column m104, it makes
% % % % % n4 number of branches
% % % % n4 = randi([2 2],1,1);  % Synapses 2-4 from Nodes
% % % % for k = 1:n4 
% % % %     ss(k) = true ;
% % % % %     ss %Test
% % % % end
% % % % New_W1d(:,nn) = ss ;
% % % % end
% % % % New_W1d;
% % % % a1 = 0;a2 = zeros(1,f);
% % % % for ii = 1:f
% % % % for jj = f+1
% % % % a1 = New_W1d(1:jj,ii);
% % % % end
% % % % a2(ii) = sum(a1);
% % % % end
% % % % m105 = max(a2);
% % % % %% 4rth Layer%Middle Matrix  1st, Same SUBSTRUCTURE Make 2 branches, 4rth Layer
% % % % % m105 = sum(New_W1c(1,1:end)) ;
% % % % for nn=1:m105    %Middle Matrix, nn = f column feature
% % % % ss = New_W1c(:,nn) ;
% % % % % Synapses 2-4 from Nodes. For each column m105, it makes
% % % % % n5 number of branches
% % % % n5 = randi([2 2],1,1);  % Synapses 2-4 from Nodes
% % % % for k = 1:n5 
% % % %     ss(k) = true ;
% % % % %     ss %Test
% % % % end
% % % % New_W1c(:,nn) = ss ;
% % % % end
% % % % New_W1c;
% % % % a1 = 0;a2 = zeros(1,f);
% % % % for ii = 1:f
% % % % for jj = f+1
% % % % a1 = New_W1c(1:jj,ii);
% % % % end
% % % % a2(ii) = sum(a1);
% % % % end
% % % % m106 = max(a2);
% % % % % %% 3rd Layer%Middle Matrix  1st, Same SUBSTRUCTURE Make 2 branches, 3rd Layer
% % % % % % m106 = sum(New_W1d(1,1:end))  ;
% % % % % for nn=1:m102    %Middle Matrix, nn = f column feature
% % % % % ss = New_W1b(:,nn) ;
% % % % % % Synapses 2-4 from Nodes. For each column m106, it makes
% % % % % % n6 number of branches
% % % % % n6 = randi([2 2],1,1);  % Synapses 2-4 from Nodes
% % % % % for k = 1:n6 
% % % % %     ss(k) = true ;
% % % % % %     ss %Test
% % % % % end
% % % % % New_W1b(:,nn) = ss ;
% % % % % end
% % % % % New_W1b;
% % % % % a1 = 0;a2 = zeros(1,f);
% % % % % for ii = 1:f
% % % % % for jj = f+1
% % % % % a1 = New_W1b(1:jj,ii);
% % % % % end
% % % % % a2(ii) = sum(a1);
% % % % % end
% % % % % m107 = max(a2);
% % % % % % %% 2nd Layer%Middle Matrix  1st, Same SUBSTRUCTURE Make 2 branches, 2nd Layer
% % % % % % % m107 = sum(New_W1e(1,1:end))  ;
% % % % % % for nn=1:m107    %Middle Matrix, nn = f column feature
% % % % % % ss = New_W1a(:,nn) ;
% % % % % % % Synapses 2-4 from Nodes. For each column m107, it makes
% % % % % % % n7 number of branches
% % % % % % n7 = randi([2 2],1,1);  % Synapses 2-4 from Nodes
% % % % % % for k = 1:n7 
% % % % % %     ss(k) = true ;
% % % % % % %     ss %Test
% % % % % % end
% % % % % % New_W1a(:,nn) = ss; 
% % % % % % end
% % % % % % New_W1a;
% % % % % % a1 = 0;a2 = zeros(1,f);
% % % % % % for ii = 1:f
% % % % % % for jj = f+1
% % % % % % a1 = New_W1a(1:jj,ii);
% % % % % % end
% % % % % % a2(ii) = sum(a1);
% % % % % % end
% % % % % % m108 = max(a2);
% 1st Layer %Middle Matrix  1st, Same SUBSTRUCTURE Make 2 branches, 1st Layer
m108 = sum(New_W1(1,1:end))   ;
for nn=1:m108 
for nn=1:f-1    %Middle Matrix, nn = f column feature
ss = New_W1(:,nn) ;
% Synapses 2-4 from Nodes. For each column m108, it makes
% n8 number of branches
% n8 = randi([1 2],1,1);  % Synapses 2-4 from Nodes
n8 = randi([2 2],1,1);  % Synapses 2-4 from Nodes
for k = 1:n8 
    ss(k) = true ;
%     ss %Test
end
New_W1(:,nn) = ss; 
end
end
New_W1;

% ouput_column_element =  d(m)
% dummy_output_column = d(m);
% Branches =  [n1 n2 n3 n4 n5 n6 n7 n8 m101]  %n8 Root branch m101 ROOT
Final_ouput_matrix102 = [New_W1 New_W2];
%% Neuron Connection
   a = Final_ouput_matrix102; 
   aa = size(Final_ouput_matrix102);
   size_a = 0;
   for k = 1:aa(1)
       for kk = 1:aa(2)
           if (a(k,kk) ~= 0)
           size_a = size_a + 1;  
           end
       end
   end
   Neuron_Connections_tree_matrix = size_a;

