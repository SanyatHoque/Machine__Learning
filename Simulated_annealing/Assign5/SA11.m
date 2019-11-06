clear; close all;
cC = load('EUC_2D_100.txt');
numCities = size(cC,1);
x=cC(1:numCities, 1);
y=cC(1:numCities, 2);
x(numCities+1)=cC(1,1);
y(numCities+1)=cC(1,2);

figure   
hold on
plot(x',y','.k','MarkerSize',14)
labels = cellstr( num2str([1:numCities]') );  %' # labels correspond to their order
text(x(1:numCities)', y(1:numCities)', labels, 'VerticalAlignment','bottom', ...
                             'HorizontalAlignment','center');
ylabel('Y Coordinate', 'fontsize', 18, 'fontname', 'Arial');
xlabel('X Coordinate', 'fontsize', 18, 'fontname', 'Arial');
title('City Coordinates', 'fontsize', 20, 'fontname', 'Arial');

numCoolingLoops = 8000;   % 20000;   
numEquilbriumLoops = 200;
pStart = 0.56;        % Probability of accepting worse solution at the start
pEnd = 0.0000001;        % Probability of accepting worse solution at the end
tStart = -1.0/log(0.56); % Initial temperature   1.7247
tEnd = -1.0/log(0.0000001);     % Final temperature     0.0620
jj = 10000;   %10000;
frac = (tEnd/tStart)^(1.0/(jj-1.0));% Fractional reduction per cycle
cityRoute_i = randperm(numCities); % Get initial route
cityRoute_b = cityRoute_i;
cityRoute_j = cityRoute_i;
cityRoute_o = cityRoute_i;
% Initial distances
D_j = computeEUCDistance(numCities, cC, cityRoute_i);
D_o = D_j; D_b = D_j; D(1) = D_j; D1(1) = D_j;  
numAcceptedSolutions = 1.0;
tCurrent = tStart;         % Current temperature = initial temperature
DeltaE_avg = 0.0;   % DeltaE Average
tic
for i=1:numCoolingLoops
for j=1:numEquilbriumLoops
        cityRoute_j = perturbRoute(numCities, cityRoute_b);
        D_j = computeEUCDistance(numCities, cC, cityRoute_j);
        DeltaE = abs(D_j-D_b);
        test1 = D_b;
        if (D_j > D_b) % objective function is worse
            if (i==1 && j==1) DeltaE_avg = DeltaE; end
            p = exp(-DeltaE/(DeltaE_avg * tCurrent));
            if (p > rand()) accept = true; else accept = false; end
        else accept = true; % objective function is better
        end
        if (accept==true)
            cityRoute_b = cityRoute_j;
            D_b = D_j;
            numAcceptedSolutions = numAcceptedSolutions + 1.0;
            DeltaE_avg = (DeltaE_avg * (numAcceptedSolutions-1.0) + ... 
                                            DeltaE) / numAcceptedSolutions;
        end
end
     test = D_b;
    %%
    if (i > 6000)    % > 11638
     stage = 2;  
numEquilbriumLoops = 700;
jj = 1000;
tStart = -1.0/log(0.56); % Initial temperature   1.7247
tEnd = -1.0/log(0.0000001);     % Final temperature     0.0620
%  frac = (tEnd/tStart)^(70/(00.999999999*jj-10));% Fractional reduction per cycle
       frac = (tEnd/tStart)^(1.0/(jj-1.0));%
%     prev_tCurrent = tStart;
%     prev_tCurrent = 1*tStart/(1+1*log(jj+1)) ; % Lower the temperature for next cycle
% frac = 1;
    else
        stage = 1;
        jj = 10000;
        tStart = -1.0/log(0.56); % Initial temperature   1.7247
        tEnd = -1.0/log(0.0000001);     % Final temperature     0.0620
        frac = (tEnd/tStart)^(1.0/(jj-1.0));% Fractional reduction per cycle
    end
    %%
    tCurrent = frac * tCurrent; % Lower the temperature for next cycle
    cityRoute_o = cityRoute_b;  % Update optimal route at each cycle
    D(i+1) = D_b; %record the route distance for each temperature setting
    D1(i+1) = D_j;
    D_o = D_b; % Update optimal distance
     disp([num2str(i),' stage: ',num2str(stage),' temperature: ',num2str(tCurrent),' D_b: ',num2str(D_b),' D_j: ',num2str(D_j)])
end
toc
D_b

figure
set(0, 'defaultaxesfontname', 'Arial');
set(0, 'defaultaxesfontsize', 14);
plot(D,'r.-')
ylabel('Distance', 'fontsize', 14, 'fontname', 'Arial');
xlabel('Route Number', 'fontsize', 14, 'fontname', 'Arial');
title('Distance vs Route Number', 'fontsize', 16, 'fontname', 'Arial');
%15160 class: 2 temperature: 0.011152D_b:2900D_j:3789.1045
hold on;
plot(D1,'b.-')
% Compute distance
D_b=0; cR = cityRoute_o;
for i=1:numCities-1
	D_b = D_b + sqrt((cC(cR(i),1)-cC(cR(i+1),1))^2 + (cC(cR(i),2)-cC(cR(i+1),2))^2);
end
D_b = D_b + sqrt((cC(cR(numCities),1)-cC(cR(1),1))^2 + (cC(cR(numCities),2)-cC(cR(1),2))^2);
disp(['Best algo   objective: ',num2str(D_b)])
disp(['Best global objective: ',num2str(D_o)])

%Save city route to file
fileID = fopen('BestCR.txt','w');
fprintf(fileID,'%6.2f\n',cR);
fclose(fileID);

hold off
figure
set(0, 'defaultaxesfontname', 'Arial');
set(0, 'defaultaxesfontsize', 14);
plot(D,'r.-')
ylabel('Distance', 'fontsize', 14, 'fontname', 'Arial');
xlabel('Route Number', 'fontsize', 14, 'fontname', 'Arial');
title('Distance vs Route Number', 'fontsize', 16, 'fontname', 'Arial');


for i=1:numCities
    x(i)=cC(cR(i),1);
    y(i)=cC(cR(i),2);
end
x(numCities+1)=cC(cR(1),1);
y(numCities+1)=cC(cR(1),2);
figure
hold on
plot(x',y',...
    'r',...
    'LineWidth',1,...
    'MarkerSize',8,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[1.0,1.0,1.0])
plot(x(1),y(1),...
    'r',...
    'LineWidth',1,...
    'MarkerSize',8,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[1.0,0.0,0.0])
labels = cellstr( num2str([1:numCities]') );  %' # labels correspond to their order
text(x(1:numCities)', y(1:numCities)', labels, 'VerticalAlignment','middle', ...
                             'HorizontalAlignment','center')

%plot(x',y','MarkerSize',24)
ylabel('Y Coordinate', 'fontsize', 18, 'fontname', 'Arial');
xlabel('X Coordinate', 'fontsize', 18, 'fontname', 'Arial');
title('Best City Route', 'fontsize', 20, 'fontname', 'Arial');


