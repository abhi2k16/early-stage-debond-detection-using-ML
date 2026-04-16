%%
clear; clear all;
x_exp_R1=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\15-09-2021\scope_13.xlsx');
x_exp_R2=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\15-09-2021\scope_14.xlsx');
%%
%%
%x_exp2(2:752,3)=x_exp2(2:752,3).*0.1;
figure
plot(x_exp_R1(3:end,1).*1e6,x_exp_R1(3:end,3)./max(x_exp_R1(3:end,3)),'k-','LineWidth',1)
hold on
plot(x_exp_R2(3:end,1).*1e6,x_exp_R2(3:end,3)./max(x_exp_R2(3:end,3)),'r-','LineWidth',1)
xlim([0 100])
ylim([-1 1])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('PZT_1','PZT_2')
%%
x_exp_R1(3:445,3)=x_exp_R1(3:445,3).*0.2;
x_exp_R2(3:600,3)=x_exp_R2(3:600,3).*0;
figure
subplot(2,1,1)
plot(x_exp_R1(3:end,1).*1e6,x_exp_R1(3:end,3)./max(x_exp_R1(3:end,3)),'b-','LineWidth',1)
xlim([0 200])
ylim([-1 1])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('PZT1')
subplot(2,1,2)
plot(x_exp_R2(3:end,1).*1e6,x_exp_R2(3:end,3)./max(x_exp_R2(3:end,3)),'r-','LineWidth',1)
xlim([0 200])
ylim([-1 1])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('PZT2')
%%
figure
for i=1:2
    if i==1
        x3=x_exp_R1(3:end,3);
    else 
        x3=x_exp_R2(3:end,3);
    end
    T_total=0.0002;
    Fs = (length(x3)-1)/T_total;  % Sampling frequency
    T = 1/Fs;                     % Sampling period
    L = length(x3);               % Length of signal
    t = (0:L-1)*T;                % Time vector
    %n = 2^nextpow2(L);
    Y=fft(x3);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    hold on
    if i==1
        plot(f./1000,P1./max(P1),'b-','LineWidth',1) 
    else
        plot(f./1000,(P1./max(P1)),'r-.','LineWidth',1)
    end
    xlim([0 500])
    ylim([0 0.1])
end
 xlabel('Frequency kHz')
 ylabel('Normalized Amplitute')
 legend('PZT_1','PZT_2')
%%