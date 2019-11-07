%Maximum Likelihood Estimate : Kernel Esttimator
%Arbitrary Variance, non-diagonal elements not equal to zero
clc;clear all; close all;
load data_bayes3
%% Plot the data
labels = [1;2;3];
markers = {'ko','kd','ks'};        %appearance of a point on the plot
color = {'magenta','green','blue'};
figure;
hold off
for c = 1:3
    position = find(y==labels(c));   %finds position of each classes on Meshgrid
    plot(X(position,1),X(position,2),markers{c},'markersize',5,'linewidth',1,'markerfacecolor',color{c});
    hold on
end
set(gca,'Title',text('String','Data, 3 Distinct Classes','FontAngle', 'italic', 'FontWeight', 'bold'),'xlabel',text('String', '$\mathbf{X_1}$', 'Interpreter', 'latex'),'ylabel',text('String', '$\mathbf{X_2}$', 'Interpreter', 'latex'))
legend('class 1','class 2','class 3')
%% Fit class-conditional Gaussians for each class
%for each class, estimate the mean and covariance matrix from the data
for c = 1:3
    position = find(y==labels(c));   %finds position of each classes on Meshgrid
    variable_1=y==labels(c);   %labels each class (eg. 1,2 or 3) with 1 and others with 0
    variable_2=(X(position,1:2)); %finds position of Meshgrid for each classes  %2 columns in X out of 90*2
    variable_3=mean(X(position,1:2));  %calculates Mean position of Meshgrid for each classes
    class_mean(c,1:2) = mean(X(position,1:2)) ;  % (2x1)
% Covariance Matrix, (2x2), Maximum Likelihood Estimate for each class c
    feature_covar(1:2,1:2,c) = cov(X(position,1:2),1); %Normalized by N (not N-1)
    variable_4=cov(X(position,1:2),1);
end
% % With Naive Bayes assumption : 2 Features are independant to each other for each class 
% for c = 1:3
%     feature_covar(1,2,c) = 0;
%     feature_covar(2,1,c) = 0;
% end
% % With assumption : 2 Features have same variance for each class 
% for c = 1:3
%     feature_covar(1,2,c) = 0;
%     feature_covar(2,1,c) = 0;
%     variable1 = feature_covar(1,1,c) ;
%     feature_covar(2,2,c) = variable1 ;
% end
%% Compute the Probability densitiy for each class 
[X1,X2] = meshgrid(0:0.2:10,0:0.2:10);
for c = 1:3
    variable_5 = class_mean(c,:);  %Mean for each classes
    variable_6 = feature_covar(:,:,c);   %Calculates Covariance for each features of classes
% Function that calculates probability_density for each points in Meshgrid
    probability_density_function = density_norm([X1(:),X2(:)], class_mean(c,:), feature_covar(:,:,c));
% reshape function converts an array of 1D to 2D meshgrid, c is no. of classes
% reshape makes 3 separate arrays to construct contours over 2 feature meshgrid
    probability_density(:,:,c) = reshape(probability_density_function,size(X1));
end
%% plot Contours of constant p(X1,X2) iso-probability contour ellipses for each class
figure
for i=1:3
    for c = 1:3
        position = find(y==labels(c));
        plot(X(position,1),X(position,2),markers{c},'markersize',5,'linewidth',1,'markerfacecolor',color{c});
        hold on
    end
% X1 is feature 1 X2 is feature 2,probability_density is 3rd variable that describes elevation of graph 
    contour(X1,X2,probability_density(:,:,i));    
    set(gca,'Title',text('String','Data with Class-conditional Iso-probability Lines','FontAngle', 'italic', 'FontWeight', 'bold'),'xlabel',text('String', '$\mathbf{X_1}$', 'Interpreter', 'latex'),'ylabel',text('String', '$\mathbf{X_2}$', 'Interpreter', 'latex'))
    legend('class 1','class 2','class 3')
end
% %% Plot the predictive contours for each class
% %normalize the probabilities first
% variable_7 = repmat(sum(probability_density,3),[1,1,3]);
% probability_density1 = probability_density./repmat(sum(probability_density,3),[1,1,3]);
% for i = 1:3 
%     figure(i+2);%figures 3 to 5
%     hold off
%     for c = 1:3
%         position = find(y==labels(c)) ;
%         variable_8 = X(position,1);% finds position of Meshgrid's X1 for each classes  
%         variable_9 = X(position,2);% finds position of Meshgrid's X1 for each classes 
%         plot(X(position,1),X(position,2),markers{c},'markersize',5,'linewidth',1,'markerfacecolor',color{c});
%         hold on
%     end
%     contour(X1,X2,probability_density1(:,:,i));
%     info = sprintf('Probability contours for class %g',i);
%     set(gca,'Title',text('String',info,'FontAngle', 'italic', 'FontWeight', 'bold'),'xlabel',text('String', '$\mathbf{X_1}$', 'Interpreter', 'latex'),'ylabel',text('String', '$\mathbf{X_2}$', 'Interpreter', 'latex'))
% end
%% Feature X1 : 1D Gaussian Normal Distribution for X1
format long
X11 = linspace(min(X1(1,1:end)), max(X1(1,1:end))); % Feature X1
norm11 = normpdf(X11,class_mean(1,1),feature_covar(1,1,1));
norm21 = normpdf(X11,class_mean(2,1),feature_covar(1,1,2));
norm31 = normpdf(X11,class_mean(3,1),feature_covar(1,1,3));
figure ;hold on;
plot(X11,norm11,'r')
plot(X11,norm21,'b')
plot(X11,norm31,'g')
xs=X11(X11>3.93939385 & X11<6); %Range of Area under Blue curved Gaussian Distribution
xs1=X11(X11>1 & X11<3.93939385); %Range of Area under Red curved Gaussian Distribution
xs2=X11(X11>5.149 & X11<9); %Range of Area under Green curved Gaussian Distribution
xs3=X11(X11>1 & X11<5.149); %Range of Area under Red curved Gaussian Distribution
figure;hold on;
x_total = X11(X11>0 & X11<10);
Area11 = area(xs,normpdf(xs,class_mean(1,1),feature_covar(1,1,1))) ;
Area21 = area(xs1,normpdf(xs1,class_mean(3,1),feature_covar(1,1,3))) ; 
Area31 = area(xs2,normpdf(xs2,class_mean(3,1),feature_covar(1,1,3))) ;
Area41 = area(xs3,normpdf(xs3,class_mean(2,1),feature_covar(1,1,2))) ;
plot(X11,normpdf(X11,class_mean(1,1),feature_covar(1,1,1)),'r')
plot(X11,normpdf(X11,class_mean(3,1),feature_covar(1,1,3)),'b')
plot(X11,normpdf(X11,class_mean(2,1),feature_covar(1,1,2)),'g')
%% Feature X2 : 1D Gaussian Normal Distribution for X2
X22 = linspace(min(X2(1:end,1)), max(X2(1:end,1))); % Feature X2
norm12 = normpdf(X22,class_mean(1,2),feature_covar(2,2,1));
norm22 = normpdf(X22,class_mean(2,2),feature_covar(2,2,2));
norm32 = normpdf(X22,class_mean(3,2),feature_covar(2,2,3));
figure ;hold on;
plot(X22,norm12,'b')
plot(X22,norm22,'g')
plot(X22,norm32,'r')
xr1=X22(X22>4.0 & X22<8); %Range of Area under Blue curved Gaussian Distribution
xr=X22(X22>0 & X22<4.0); %Range of Area under Red curved Gaussian Distribution
xr2=X22(X22>6.0599989999 & X22<9); %Range of Area under Blue curved Gaussian Distr
xr3=X22(X22>1 & X22<6.0599989999); %Range of Area under Blue curved Gaussian Distr
xr4=X22(X22>6.0599989999 & X22<10);  %Range of Area under Green curved Gaussian Distr
figure;hold on;
Area12 = area(xr,normpdf(xr,class_mean(1,2),feature_covar(2,2,1))) ;
Area22 = area(xr1,normpdf(xr1,class_mean(3,2),feature_covar(2,2,3))) ;
Area32 = area(xr2,normpdf(xr2,class_mean(3,2),feature_covar(2,2,3))) ;
Area42 = area(xr3,normpdf(xr3,class_mean(2,2),feature_covar(2,2,2))) ;
Area52 = area(xr4,normpdf(xr4,class_mean(1,2),feature_covar(2,2,1))) ;
plot(X22,normpdf(X22,class_mean(1,2),feature_covar(2,2,1)),'b')
plot(X22,normpdf(X22,class_mean(3,2),feature_covar(2,2,3)),'r')
plot(X22,normpdf(X22,class_mean(2,2),feature_covar(2,2,2)),'g')
% %% plot the surface in 3D in Meshgrid
% for c = 1:3
%  % Function that calculates probability_density for each points in Meshgrid
%     probability_density_function = density_norm([X1(:),X2(:)], class_mean(c,:), feature_covar(:,:,c));
% % reshape function converts an array of 1D to 2D/3D meshgrid, c is no. of classes
% % reshape makes 3 separate arrays to construct contours over 2 feature meshgrid
%     probability_density(:,:,c) = reshape(probability_density_function,size(X1));
%     mesh(X1, X2, probability_density(:,:,c));
% end
% % Add title and axis labels
% set(gca,'Title',text('String','A 3-D View of the Bivariate Normal Density',...
% 'FontAngle', 'Italic', 'FontWeight', 'bold'),...
% 'xlabel',text('String', '$\mathbf{X}$', 'Interpreter', 'latex'),...
% 'ylabel',text('String', '$\mathbf{Y}$', 'Interpreter', 'latex'),...
% 'zlabel',text('String', 'density', 'FontAngle', 'Italic', 'FontWeight', 'bold'))
% view(-10, 50)
% colormap('hot')
%% Testing 
x = [4 7];   %Testing Observation Vector
%  class_mean(class,:);
%  feature_covar(:,:,class);
% using Parameter Estimation using Mean and Covariance of a Feature Distribution
% if Featurres X1, X2 are independant, Covariance with non zero non-diagonal elements are used.
% if Featurres X1, X2 are dependant, Variance with null components of non-diagonal elements are used.
for class = 1:3
    aposterior(class) = density_norm(x, class_mean(class,:), feature_covar(:,:,class))
end
% %% OR Training each pdf for each feature X1, X2 and using : CHAIN RULE if Bayes'Assumption is taken
% % CHAIN RULE using Bayesian Conditional Independance
% % Using Bayesian Conditional Independance and finding X using MLE gives
% % same value if X1, X2 are independant 
% for c = 1:3
%     for feature = 1:2
%     test101 = class_mean(c,feature);
%     test102 = feature_covar(feature,feature,c);    
%     maximum_aposterior1(c,feature) = density_norm2(x(feature), class_mean(c,feature), feature_covar(feature,feature,c));
%     end
%     maximum_aposterior(c) = maximum_aposterior1(c,1)*maximum_aposterior1(c,2); 
% end
% maximum_aposterior
%%
% Weak Prior Belief probability (i.e., uniform prior), MAP gives the same result as MLE.
%Labelled Data for Class 1 = 30; %Labelled Data for Class 2 = 30;%Labelled Data for Class 3 = 30;
%Belief Probability for each Class (Uniform Priors) = 30/90;(Since,Total Data X = 90) = 1/3 ;
aposterior = ((1/3)*aposterior) ;%maximum posterior probability
maximum_aposterior = max(aposterior) %maximum posterior probability
find(maximum_aposterior == aposterior)  %index class of maximum aposterior 