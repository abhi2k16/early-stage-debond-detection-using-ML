%%
clear; clear all;
x_01=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_8x16mm.rpt');
%x_02=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_10x14mm.rpt');
x_03=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_10x16mm.rpt');
x_04=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_12x16mm.rpt');
x_05=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam_Category\U2_at_centre_14x16mm.rpt');
%% FFT
n=2;
X_dam=[x_01.data(:,n) x_03.data(:,n) x_04.data(:,n) x_05.data(1:10001,2)];% x_05.data(1:10001,2)];
T_total = 0.0001;
Fs = (length(X_dam(1:end,1)))/T_total;  % Sampling frequency
T = 1/Fs;                             % Sampling period
L = length(X_dam(1:end,1));             % Length of signal
t = (0:L-1)*T;                        % Time vector
for i=1:4
    Y=fft(X_dam(:,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    P_B(:,i)=P1;
    F(:,i)=f;
end
%% Ploting FFT
figure
lw = 1.5;
plot(F(:,1)./1000,P_B(:,1)./max(P_B(:,1)),'r-.','LineWidth',lw) 
hold on
plot(F(:,2)./1000,P_B(:,2)./max(P_B(:,2)),'b-.','LineWidth',lw) 
plot(F(:,3)./1000,P_B(:,3)./max(P_B(:,3)),'c-','LineWidth',lw) 
plot(F(:,4)./1000,P_B(:,4)./max(P_B(:,4)),'m-.','LineWidth',lw) 
%plot(F(:,5)./1000,P_B(:,5)./max(P_B(:,5)),'k-.','LineWidth',lw) 
xlim([0 800])
ylim([0 0.2])
xlabel('Frequency (kHz)')
ylabel('Normalized Amplitude')
legend('Centre','Centre-Edge','Edge','Thr-Width','Location','northeast')
%% Ploting FFT in db
figure
lw = 2;
plot(F(:,1)./1000,db(P_B(:,1)./max(P_B(:,1))),'r-.','LineWidth',lw) 
hold on
plot(F(:,2)./1000,db(P_B(:,2)./max(P_B(:,2))),'b-.','LineWidth',lw) 
plot(F(:,3)./1000,db(P_B(:,3)./max(P_B(:,3))),'k-','LineWidth',lw) 
plot(F(:,4)./1000,db(P_B(:,4)./max(P_B(:,4))),'m-.','LineWidth',lw) 
%plot(F(:,5)./1000,db(P_B(:,5)./max(P_B(:,5))),'g-.','LineWidth',lw) 
xlim([0 900])
ylim([-70 0])
xlabel('Frequency (kHz)')
ylabel('Normalized Amplitude(db)')
legend('Centre','Centre-Edge','Edge','Thr-Width','Location','northeast')
%% FFT of split signal segment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=2;
X_dam=[x_01.data(:,n) x_03.data(:,n) x_04.data(:,n) x_05.data(1:10001,2)];% x_05.data(1:10001,2)];
% Lenght of split segmented signal
l1=5000; l2=7001;
T_total = 0.00002;
Fs = (length(X_dam(l1:l2,1)))/T_total;  % Sampling frequency
T = 1/Fs;                             % Sampling period
L = length(X_dam(l1:l2,1));             % Length of signal
t = (0:L-1)*T;                        % Time vector
for i=1:4
    Y=fft(X_dam(l1:l2,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    P_B(:,i)=P1;
    F(:,i)=f;
end
%% Ploting FFT
figure
lw = 1.5;
plot(F(:,1)./1000,P_B(:,1)./max(P_B(:,1)),'r-.','LineWidth',lw) 
hold on
plot(F(:,2)./1000,P_B(:,2)./max(P_B(:,2)),'b-.','LineWidth',lw) 
plot(F(:,3)./1000,P_B(:,3)./max(P_B(:,3)),'c-','LineWidth',lw) 
plot(F(:,4)./1000,P_B(:,4)./max(P_B(:,4)),'m-.','LineWidth',lw) 
%plot(F(:,5)./1000,P_B(:,5)./max(P_B(:,5)),'k-.','LineWidth',lw) 
xlim([0 800])
%ylim([0 0.2])
xlabel('Frequency (kHz)')
ylabel('Normalized Amplitude')
legend('Centre','Centre-Edge','Edge','Thr-Width','Location','northeast')
%% Ploting FFT in db
figure
lw = 2;
plot(F(:,1)./1000,db(P_B(:,1)./max(P_B(:,1))),'r-.','LineWidth',lw) 
hold on
plot(F(:,2)./1000,db(P_B(:,2)./max(P_B(:,2))),'b-.','LineWidth',lw) 
plot(F(:,3)./1000,db(P_B(:,3)./max(P_B(:,3))),'k-','LineWidth',lw) 
plot(F(:,4)./1000,db(P_B(:,4)./max(P_B(:,4))),'m-.','LineWidth',lw) 
%plot(F(:,5)./1000,db(P_B(:,5)./max(P_B(:,5))),'g-.','LineWidth',lw) 
%xlim([0 900])
%ylim([-70 0])
xlabel('Frequency (kHz)')
ylabel('Normalized Amplitude(db)')
legend('Centre','Centre-Edge','Edge','Thr-Width','Location','northeast')
%%