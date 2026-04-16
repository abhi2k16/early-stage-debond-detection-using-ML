%%
clear; clear all;
%% data collection centre
X_centre = zeros(10002,25);
for i = 1:5
    for j = 1:5
        dam_cat = 'U2_at_centre_';
        file = append(num2str(8+(i-1)*2),'x',num2str(8+(j-1)*2),'mm','.','rpt');
        D_area_cen((i-1)*5+j) = (8+(i-1)*2)*(8+(j-1)*2);
        file_name= append('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\',dam_cat,file);
        x = importdata(file_name).data(:,2);
        X_centre(1:10001,(i-1)*5+j) = x;
    end
end
X_centre(10002,1:25)=D_area_cen;
%xlswrite('U2_at_centre_8x8_to_16x16.xlsx',X_centre);
%% data collection for edge
X_edge = zeros(10002,10);
for i = 1:5
    for j = 1:2
        dam_cat = 'U2_at_edge_';
        file = append(num2str(8+(i-1)*2),'x',num2str(11+(j-1)*1),'mm','.','rpt');
        D_area_edg((i-1)*2+j) = (8+(i-1)*2)*(11+(j-1)*1);
        file_name= append('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\',dam_cat,file);
        x_e = importdata(file_name).data(:,2);
        X_edge(1:10001,(i-1)*2+j) = x_e;
    end
end
X_edge(10002,1:10)=D_area_edg;
%xlswrite('U2_at_edge_8x11_to_16x12.xlsx',X_edge);
%% data collection for centre_edge
X_centre_edge = zeros(10002,15);
for i = 1:5
    for j = 1:3
        dam_cat = 'U2_at_centre_edge_';
        file = append(num2str(8+(i-1)*2),'x',num2str(16+(j-1)*1),'mm','.','rpt');
        D_area_cen_edg((i-1)*3+j) = (8+(i-1)*2)*(16+(j-1)*1);
        file_name= append('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\',dam_cat,file);
        x_e = importdata(file_name).data(:,2);
        X_centre_edge(1:10001,(i-1)*3+j) = x_e;
    end
end
X_centre_edge(10002,1:15)=D_area_cen_edg;
%xlswrite('U2_at_centre_edge_8x16_to_16x18.xlsx',X_centre_edge);
%% data collection for through_the_width
X_thr_width = zeros(10002,9);
for i = 1:9
    for j = 1:1
        dam_cat = 'U2_thr_width_';
        file = append(num2str(8+(i-1)*1),'x',num2str(26+(j-1)*1),'mm','.','rpt');
        D_area_thr_wid((i-1)*1+j) = (8+(i-1)*1)*(26+(j-1)*1);
        file_name= append('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\',dam_cat,file);
        x_e = importdata(file_name).data(:,2);
        X_thr_width(1:10001,(i-1)*1+j) = x_e;
    end
end
X_thr_width(10002,1:9) = D_area_thr_wid;
%xlswrite('U2_thr_width_8x26_to_16x26.xlsx',X_thr_width);
%% FFT of split signal segment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_dam= X_centre;
% Lenght of split segmented signal
l1=5001; l2=10001;
T_total = 0.00005;
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
%% Plot FFT 
figure
for k=1:25
    hold on
    lw = 1.5;
    plot(0:(Fs/n)./1000:(Fs/2-Fs/n)./1000,db(P_centre(:,k)./max(P_centre(:,k))),'r-.','LineWidth',lw)
    xlim([0 800])
    %ylim([0 0.2])
end
%% FFT for x_edge
X_dam= X_edge;                 
for i=1:10
    Y=fft(X_dam(l1:l2,i)',n);
    P2 = abs(Y/L);
    P1 = P2(:,1:n/2+1);
    P1(:,2:end-1) = 2*P1(:,2:end-1);
    f = Fs*(0:(L/2))/L;
    P_edge(:,i)=P1(:,1:n/2);
    F(:,i)=f;
end
%% plot FFT
figure
for k=1:10
    hold on
    lw = 1.5;
    plot(0:(Fs/n)./1000:(Fs/2-Fs/n)./1000,db(P_edge(:,k)./max(P_edge(:,k))),'b-.','LineWidth',lw) 
    xlim([0 800])
    %ylim([0 0.2])
end
%% FFT for X_centre_edge
X_dam= X_centre_edge;                      
for i=1:15
    Y=fft(X_dam(l1:l2,i)',n);
    P2 = abs(Y/L);
    P1 = P2(:,1:n/2+1);
    P1(:,2:end-1) = 2*P1(:,2:end-1);
    f = Fs*(0:(L/2))/L;
    P_centre_edge(:,i)=P1(:,1:n/2);
    F(:,i)=f;
end
%% Plot FFT 
figure
for k=1:15
    hold on
    lw = 1.5;
    plot(0:(Fs/n)./1000:(Fs/2-Fs/n)./1000,db(P_centre_edge(:,k)./max(P_centre_edge(:,k))),'b-.','LineWidth',lw) 
    xlim([0 800])
    %ylim([0 0.2])
end
%% FFT for X_centre_edge
X_dam= X_thr_width;
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
for k=1:9
    hold on
    lw = 1.5;
    plot(0:(Fs/n)./1000:(Fs/2-Fs/n)./1000,db(P_thr_width(:,k)./max(P_thr_width(:,k))),'m-.','LineWidth',lw) 
    xlim([0 800])
    %ylim([0 0.2])
end
%% RMSD of FFT_signals
RMSD = sqrt(((sum(P_thr_width')-sum(P_centre_edge')).^2)./sum(P_thr_width').^2);
%% 
figure
plot(0:(Fs/n)./1000:(Fs/2-Fs/n)./1000,RMSD,'b.-','LineWidth',1.5)
xlim([0 800])
xlabel('Frequency (kHz)')
ylabel('RMSD')
title('Full Vs Centre-n-Edge')
%%

