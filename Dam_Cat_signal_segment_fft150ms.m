%%
clear; clear all;
X_centre = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_8x8_to_16x16_150ms.xlsx");
X_centre_edge = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_edge_8x15_to_16x18_150ms.xlsx");
X_edge = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_edge_8x10_to_20x12_150ms.xlsx");
X_thr_width = xlsread("C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_thr_width_8x26_to_16x26_150ms.xlsx");
%% FFT of split signal segment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
snr = 100;
X_dam=awgn(rescale(X_centre(1:15001,:),-1,1),snr);
x_t1 = X_centre(15002,:);  % Regression target vector
% Lenght of split segmented signal
l1=1; l2=10001;
T_total = 0.0001;
L = length(X_dam(l1:l2,1));           % Length of signal
Fs = (L-1)/T_total;                   % Sampling frequency
T = 1/Fs;                             % Sampling period
n = 2^nextpow2(L/2);                    % the n-point DFT
t = (0:L-1)*T;                        % Time vector
for m = 1:4
    for i=1:25
        if m==1
            l1=1;l2=10001;
            Y=fft(X_dam(l1:l2,i)',n);
        elseif m == 2
            l1=1001;l2=11001;
            Y=fft(X_dam(l1:l2,i)',n);
        elseif m == 3
            l1=2001;l2=12001;
            Y=fft(X_dam(l1:l2,i)',n);
        else
            l1=3001;l2=13001;
            Y=fft(X_dam(l1:l2,i)',n);
        end
        P2 = abs(Y/L);
        %P1 = P2(1:L/2+1);
        %P1(2:end-1) = 2*P1(2:end-1);
        P1 = P2(:,1:n/2+1);
        P1(:,2:end-1) = 2*P1(:,2:end-1);
        f = Fs*(0:(L/2))/L;
        %f = Fs*(0:(n/2))/n;
        P_centre(:,(m-1)*25+i)=P1(:,1:n/2);
        %F(:,i)=f;
    end
end
%% FFT for x_edge
X_dam= awgn(rescale(X_edge(1:10001,:),-1,1),snr); 
x_t2 = X_edge(10002,:);   % Regression target vector
for m = 1:3
    for i=1:10
        if m==1
            l1=1;l2=8001;
            Y=fft(X_dam(l1:l2,i)',n);
        elseif m == 2
            l1=1001;l2=9001;
            Y=fft(X_dam(l1:l2,i)',n);
        else
            l1=2001;l2=10001;
            Y=fft(X_dam(l1:l2,i)',n);
        end
        P2 = abs(Y/L);
        P1 = P2(:,1:n/2+1);
        P1(:,2:end-1) = 2*P1(:,2:end-1);
        f = Fs*(0:(L/2))/L;
        P_edge(:,(m-1)*10+i)=P1(:,1:n/2);
    end
end
%% FFT for X_centre_edge
X_dam= awgn(rescale(X_centre_edge(1:10001,:),-1,1),snr);   
x_t3 = X_centre_edge(10002,:);   % Regression target vector
for m = 1:3
    for i=1:15
        if m==1
            l1=1;l2=8001;
            Y=fft(X_dam(l1:l2,i)',n);
        elseif m == 2
            l1=1001;l2=9001;
            Y=fft(X_dam(l1:l2,i)',n);
        else
            l1=2001;l2=10001;
            Y=fft(X_dam(l1:l2,i)',n);
        end
        P2 = abs(Y/L);
        P1 = P2(:,1:n/2+1);
        P1(:,2:end-1) = 2*P1(:,2:end-1);
        f = Fs*(0:(L/2))/L;
        P_centre_edge(:,(m-1)*15+i)=P1(:,1:n/2);
    end
end
%% FFT for X_thr_width
X_dam=awgn(rescale(X_thr_width(1:10001,:),-1,1),snr);
x_t4 = X_thr_width(10002,:);  % Regression target vector
for m = 1:3
    for i=1:9
        if m==1
            l1=1;l2=8001;
            Y=fft(X_dam(l1:l2,i)',n);
        elseif m == 2
            l1=1001;l2=9001;
            Y=fft(X_dam(l1:l2,i)',n);
        else
            l1=2001;l2=10001;
            Y=fft(X_dam(l1:l2,i)',n);
        end
        P2 = abs(Y/L);
        P1 = P2(:,1:n/2+1);
        P1(:,2:end-1) = 2*P1(:,2:end-1);
        f = Fs*(0:(L/2))/L;
        P_thr_width(:,(m-1)*9+i)=P1(:,1:n/2);
    end
end
%% Plot FFT 
figure
Freq = 0:(Fs/n)./1000:(Fs/2-Fs/n)./1000;
for k=1:3
    hold on
    lw = 1.5;
    plot(Freq,db(P_centre(:,k)./max(P_centre(:,k))),'r-.','LineWidth',lw)
    plot(Freq,db(P_centre(:,25+k)./max(P_centre(:,25+k))),'b-.','LineWidth',lw)
    plot(Freq,db(P_centre(:,50+k)./max(P_centre(:,50+k))),'g-.','LineWidth',lw)
    plot(Freq,db(P_centre(:,75+k)./max(P_centre(:,75+k))),'m-.','LineWidth',lw)
    xlim([0 700])
    %ylim([0 0.2])
end
%% RMSD of FFT_signals
RMSD = sqrt(((sum(P_centre')-sum(P_thr_width')).^2)./sum(P_centre').^2);
%% 
figure
plot(0:(Fs/n)./1000:(Fs/2-Fs/n)./1000,RMSD./max(RMSD),'b-','LineWidth',1.5)
xlim([0 700])
xlabel('Frequency (kHz)')
ylabel('Avg. RMSD')
title('Centre+edge Vs Edge')
%% feature_extraction and normalization
F_centre = rescale(db(P_centre(1:41,:)'),-1,1);
F_edge = rescale(db(P_edge(1:41,:)'),-1,1);
F_centre_edge= rescale(db(P_centre_edge(1:41,:)'),-1,1);
F_thr_width = rescale(db(P_thr_width(1:41,:)'),-1,1);
X_Feature = [F_centre; F_edge; F_centre_edge; F_thr_width];
%% Normalization of feature
X_norm_feature = abs(db(X_Feature./max(X_Feature)))./max(abs(db(X_Feature./max(X_Feature))));
sigma_sample= cov(X_norm_feature); % Covarience of sample
Mu_sample = mean(X_norm_feature); % Mean of sample
%% Targer regression
T_reg = [x_t1 x_t1 x_t1 x_t2 x_t2 x_t2 x_t3 x_t3 x_t3 x_t4 x_t4 x_t4];
X_Reg = rescale(T_reg',0,1);
%% Target Class Vector
class = zeros(4,177);
for i=1:1
    for j= 1:177
        if j <= 75
            class(i,j) = 1;
        elseif j>75 & j<= 105
            class(i*2,j) = 1;
        elseif j > 105 & j <= 150
            class(i*3,j) = 1;
        else
            class(i*4,j) = 1;
        end
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
n = 2^nextpow2(L/4);                  % the n-point DFT
t = (0:L-1)*T;                        % Time vector
Y_exp=fft(X_dam_exp',n);
P2_exp = abs(Y_exp/L);
P1_exp = P2_exp(:,1:n/2+1);
P1_exp(:,2:end-1) = 2*P1_exp(:,2:end-1);
f = Fs*(0:(L/2))/L;
Freq_exp= 0:(Fs/n)./1000:(Fs/2-Fs/n)./1000;
%% Exp_feature 
F_exp = zeros(1,41);
F_exp = rescale(db(P1_exp(:,1:41)),0,1);
T_exp = logical([0,0,0,1]);
%%