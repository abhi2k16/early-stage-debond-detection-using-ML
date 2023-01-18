%%
X_FrVsAmp=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\FreqVsAmp_F00_L00.xlsx');
%%
x_freq = X_FrVsAmp(:,1);
x_F00 = X_FrVsAmp(:,2)/max(X_FrVsAmp(:,3));
x_L00 = X_FrVsAmp(:,3)/max(X_FrVsAmp(:,3));
figure;
plot(x_freq,x_F00,'b-s','LineWidth',2)
hold on
plot(x_freq,x_L00,'r-*','LineWidth',2)
xlabel('Frequency (kHz)')
ylabel('Normalized Amplitude')
legend('A0','S0')