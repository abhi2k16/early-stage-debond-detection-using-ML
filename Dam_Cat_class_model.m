%%
clear; clear all;
X_centre = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_8x8_to_16x16.xlsx");
X_centre_edge = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_edge_8x16_to_16x18.xlsx");
X_edge = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_edge_8x11_to_16x12.xlsx");
X_thr_width = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_thr_width_8x26_to_16x26.xlsx");
%% FFT of split signal segment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
snr = 100;
X_dam=awgn(rescale(X_centre,-1,1),snr);
% Lenght of split segmented signal
l1=1; l2=10001;
T_total = 0.0001;
L = length(X_dam(l1:l2,1));           % Length of signal
Fs = (L-1)/T_total;                   % Sampling frequency
T = 1/Fs;                             % Sampling period
n = 2^nextpow2(L);                    % the n-point DFT
t = (0:L-1)*T;                        % Time vector
for i=1:25
    Y=fft(X_dam(l1:l2,i)',n);
    P2 = abs(Y/L);
    %P1 = P2(1:L/2+1);
    %P1(2:end-1) = 2*P1(2:end-1);
    P1 = P2(:,1:n/2+1);
    P1(:,2:end-1) = 2*P1(:,2:end-1);
    f = Fs*(0:(L/2))/L;
    %f = Fs*(0:(n/2))/n;
    P_centre(:,i)=P1(:,1:n/2);
    F(:,i)=f;
end
%% FFT for x_edge
X_dam= awgn(rescale(X_edge,-1,1),snr);               
for i=1:10
    Y=fft(X_dam(l1:l2,i)',n);
    P2 = abs(Y/L);
    P1 = P2(:,1:n/2+1);
    P1(:,2:end-1) = 2*P1(:,2:end-1);
    f = Fs*(0:(L/2))/L;
    P_edge(:,i)=P1(:,1:n/2);
    F(:,i)=f;
end
%% FFT for X_centre_edge
X_dam= awgn(rescale(X_centre_edge,-1,1),snr);                  
for i=1:15
    Y=fft(X_dam(l1:l2,i)',n);
    P2 = abs(Y/L);
    P1 = P2(:,1:n/2+1);
    P1(:,2:end-1) = 2*P1(:,2:end-1);
    f = Fs*(0:(L/2))/L;
    P_centre_edge(:,i)=P1(:,1:n/2);
    F(:,i)=f;
end
%% FFT for X_thr_width
X_dam=awgn(rescale(X_thr_width,-1,1),snr);
for i=1:9
    Y=fft(X_dam(l1:l2,i)',n);
    P2 = abs(Y/L);
    P1 = P2(:,1:n/2+1);
    P1(:,2:end-1) = 2*P1(:,2:end-1);
    f = Fs*(0:(L/2))/L;
    P_thr_width(:,i)=P1(:,1:n/2);
    F(:,i)=f;
end
%% Plot FFT 
figure
Freq = 0:(Fs/n)./1000:(Fs/2-Fs/n)./1000;
for k=1:2
    hold on
    lw = 1.5;
    plot(Freq,db(P_centre(:,k)./max(P_centre(:,k))),'r-.','LineWidth',lw)
    xlim([0 700])
    %ylim([0 0.2])
end
%% RMSD of FFT_signals
RMSD = sqrt(((sum(P_centre_edge')-sum(P_thr_width')).^2)./sum(P_centre').^2);
%% 
figure
plot(0:(Fs/n)./1000:(Fs/2-Fs/n)./1000,RMSD./max(RMSD),'b-','LineWidth',1.5)
xlim([0 700])
xlabel('Frequency (kHz)')
ylabel('Avg. RMSD')
title('Centre+edge Vs Edge')
%% feature_extraction and normalization
F_centre = rescale(db(P_centre(1:80,:)'),-1,1);
F_edge = rescale(db(P_edge(1:80,:)'),-1,1);
F_centre_edge= rescale(db(P_centre_edge(1:80,:)'),-1,1);
F_thr_width = rescale(db(P_thr_width(1:80,:)'),-1,1);
X_Feature = [F_centre; F_edge; F_centre_edge; F_thr_width];
%% Normalization of feature
X_norm_feature = abs(db(X_Feature./max(X_Feature)))./max(abs(db(X_Feature./max(X_Feature))));
sigma_sample= cov(X_norm_feature); % Covarience of sample
Mu_sample = mean(X_norm_feature); % Mean of sample
%% Simulate Data from a Mixture of Gaussian Distributions
F_1 = F_centre;
mu1 = mean(F_1); sigma1 = cov(F_1);
F_simul_1 = mvnrnd(mu1, sigma1, 50);
F_2 = F_edge;
mu2 = mean(F_2); sigma2 = cov(F_2);
F_simul_2 = mvnrnd(mu2, sigma2, 50);
F_3 = F_centre_edge;
mu3 = mean(F_3); sigma3 = cov(F_3);
F_simul_3 = mvnrnd(mu3, sigma3, 50);
F_4 = F_thr_width;
mu4 = mean(F_4); sigma4 = cov(F_4);
F_simul_4 = mvnrnd(mu4, sigma4, 50);
X_Simul = [F_simul_1; F_simul_2; F_simul_3; F_simul_4];
%%
class = zeros(4,50);
for i=1:4
    for j= 1:50
        class(i,(i-1)*50+j) = i;
    end
end
target = class';
TARGET = logical(target);
%% EXP 
x_exp_dam=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\30-08-2021\scope_1.xlsx');
%% FFT exp data 
X_dam_exp = x_exp_dam(3:end,3);
T_total = 0.0002;
L = length(X_dam_exp);                % Length of signal
Fs = (L-1)/T_total;                   % Sampling frequency
T = 1/Fs;                             % Sampling period
n = 2^nextpow2(L/2);                  % the n-point DFT
t = (0:L-1)*T;                        % Time vector
Y_exp=fft(X_dam_exp',n);
P2_exp = abs(Y_exp/L);
P1_exp = P2_exp(:,1:n/2+1);
P1_exp(:,2:end-1) = 2*P1_exp(:,2:end-1);
f = Fs*(0:(L/2))/L;
Freq_exp= 0:(Fs/n)./1000:(Fs/2-Fs/n)./1000;
%% Exp_feature 
F_exp = zeros(1,80);
F_exp = rescale(db(P1_exp(:,1:80)),0,1);
T_exp = logical([0,0,0,1]);
%%