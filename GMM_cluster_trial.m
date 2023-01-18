%% Simulate Data from a Mixture of Gaussian Distributions

rng('default')  % For reproducibility
mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
X = [mvnrnd(mu1,sigma1,200); mvnrnd(mu2,sigma2,100)];
n = size(X,1);

figure
scatter(X(:,1),X(:,2),10,'ko')
%% Fit a Gaussian Mixture Model to the Simulated Data
options = statset('Display','final'); 
gm = fitgmdist(X,2,'Options',options)
%% Plot
hold on
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm,[x0,y0]),x,y);
fcontour(gmPDF,[-8,6])
title('Scatter Plot and Fitted GMM Contour')
hold off
%% Cluster the Data Using the Fitted GMM
idx = cluster(gm,X);
cluster1 = (idx == 1); % |1| for cluster 1 membership
cluster2 = (idx == 2); % |2| for cluster 2 membership

figure
gscatter(X(:,1),X(:,2),idx,'rb','+o')
legend('Cluster 1','Cluster 2','Location','best')
%% 