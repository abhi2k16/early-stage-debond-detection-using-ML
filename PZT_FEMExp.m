%%
clear all; clear;
x_fem=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\UnDamPZT\U1S0PDL140kHzSEN150mmUnDam.rpt');
x_fem_dam=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam10mm\U1S0PDL140kHzSEN150mmDam10mm.rpt');
x_exp_undam=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\26-08-2021\scope_4.xlsx');
x_exp_dam=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\30-08-2021\scope_1.xlsx');
%%
%x_exp2(2:752,3)=x_exp2(2:752,3).*0.1;
figure
n=2;
N=18025;
plot(x_fem.data(1:N,1).*1e6,(x_fem.data(1:N,n)./max(x_fem.data(1:10001,n))),'r-.','LineWidth',1.5)
hold on
%plot(x_fem_dam.data(1:N,1).*1e6,x_fem_dam.data(1:N,n)./max(x_fem_dam.data(1:N,n)),'r-','LineWidth',1.5)
% hold on
%plot(x_fem_dam.data(9000:N,1).*1e6,x_fem_dam.data(9000:N,n)./max(x_fem_dam.data(9000:N,n))*.3,'r-','LineWidth',1.5)
%plot(x_exp_undam(3:end,1).*1e6,x_exp_undam(3:end,3)./max(x_exp_undam(3:end,3)),'k-.','lineWidth',1.5)
plot(x_exp_dam(3:end,1).*1e6,x_exp_dam(3:end,3)./max(x_exp_dam(3:end,3))*-1,'k-.','lineWidth',1)
xlim([0 200])
ylim([-1 1])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('FEM','EXP.')
%%
%x_fft=[x_fem_dam.data(:,2) x_exp_dam(:,3)];
figure
for i=1:2
    if i==1
        x3=x_fem.data(1:18025,2);
        T_total = 0.0002;
        L = length(x3);           % Length of signal
        Fs = (L-1)/T_total;                   % Sampling frequency
        T = 1/Fs;                             % Sampling period
        n = 2^nextpow2(L);                    % the n-point DFT
        t = (0:L-1)*T; 
        Y=fft(x3',n);
        P2 = abs(Y/L);
        %P1 = P2(1:L/2+1);
        %P1(2:end-1) = 2*P1(2:end-1);
        P1 = P2(:,1:n/2+1);
        P1(:,2:end-1) = 2*P1(:,2:end-1);
    else 
        x3=x_exp_undam(3:5001,3);
        T_total = 0.0002;
        L = length(x3);           % Length of signal
        Fs = (L-1)/T_total;                   % Sampling frequency
        T = 1/Fs;                             % Sampling period
        n = 2^nextpow2(L);                    % the n-point DFT
        t = (0:L-1)*T;
        Y=fft(x3',n);
        P2 = abs(Y/L);
        %P1 = P2(1:L/2+1);
        %P1(2:end-1) = 2*P1(2:end-1);
        P1 = P2(:,1:n/2+1);
        P1(:,2:end-1) = 2*P1(:,2:end-1);
    end
%     T_total=0.0002;
%     Fs = (length(x3)-1)/T_total;  % Sampling frequency
%     T = 1/Fs;                     % Sampling period
%     L = length(x3);               % Length of signal
%     t = (0:L-1)*T;                % Time vector
%     %n = 2^nextpow2(L);
%     Y=fft(x3);
%     P2 = abs(Y/L);
%     P1 = P2(1:L/2+1);
%     P1(2:end-1) = 2*P1(2:end-1);
%     f = Fs*(0:(L/2))/L;
    hold on
    if i==1
        Freq = 0:(Fs/n)./1000:(Fs/2-Fs/n)./1000;
        plot(Freq, db(P1(:,length(Freq))./max(P1(:,length(Freq)))),'b-','LineWidth',2) 
    else
        Freq = 0:(Fs/n)./1000:(Fs/2-Fs/n)./1000;
        plot(Freq, db(P1(:,length(Freq))./max(P1(:,length(Freq)))),'r-.','LineWidth',2)
    end
    xlim([0 900])
    %ylim([0 0.1])
end
 xlabel('Frequency kHz')
 ylabel('Normalized Amplitute(db)')
 legend('FEM','Exp')
%% 