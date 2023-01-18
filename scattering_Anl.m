%%
clear; clear all;
x_00=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Scatter_Anl\U1B30F30A55_Scat_Anl_dam00mm140kHz.rpt');
x_02=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Scatter_Anl\U1B30F30A55_Scat_Anl_dam02mm140kHz.rpt');
x_04=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Scatter_Anl\U1B30F30A55_Scat_Anl_dam04mm140kHz.rpt');
x_06=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Scatter_Anl\U1B30F30A55_Scat_Anl_dam06mm140kHz.rpt');
x_08=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Scatter_Anl\U1B30F30A55_Scat_Anl_dam08mm140kHz.rpt');
x_10=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Scatter_Anl\U1B30F30A55_Scat_Anl_dam10mm140kHz.rpt');
x_12=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Scatter_Anl\U1B30F30A55_Scat_Anl_dam12mm140kHz.rpt');
%% Plotting Of Scattered Signal
n=3;
x_diff=x_00.data(1:end,n);%-x_00.data(1:end,n);
figure
plot(x_00.data(:,1).*1e6,x_00.data(1:end,n)./max(x_00.data(:,n)),'r-','LineWidth',1.5)
hold on
plot(x_02.data(:,1).*1e6,x_08.data(1:end,n)./max(x_08.data(:,n)),'b-.','LineWidth',1.5)
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('Intact','Dedonding','Location','northwest')
%% Plotting Of Scattered Signal
n=2;
figure
plot(x_00.data(:,1).*1e6,x_00.data(:,n)./max(x_00.data(:,n)),'r-','LineWidth',1)
hold on
plot(x_00.data(:,1).*1e6,x_04.data(:,n)./max(x_04.data(:,n)),'b-.','LineWidth',1)
xlabel('Time (? sec)')
ylabel('Normalized Amplitude')
legend('Intact','4mm Debonding','Location','northeast')
%%
n=3;
X_dam=[x_02.data(:,n) x_04.data(:,n) x_06.data(:,n) x_08.data(:,n) x_10.data(:,n) x_12.data(:,n)];
T_total = 0.0001;
Fs = (length(X_dam(1:end,1)))/T_total;  % Sampling frequency
T = 1/Fs;                             % Sampling period
L = length(X_dam(1:end,1));             % Length of signal
t = (0:L-1)*T;                        % Time vector
for i=1:6
    Y=fft(X_dam(:,i));
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    P_F(:,i)=P1;
    F(:,i)=f;
end
%% Ploting FFT
figure
lw = 1;
plot(F(:,1)./1000,P_F(:,1)./max(P_F(:,1)),'r-.','LineWidth',lw) 
hold on
plot(F(:,2)./1000,P_F(:,2)./max(P_F(:,2)),'b-.','LineWidth',lw) 
plot(F(:,3)./1000,P_F(:,3)./max(P_F(:,3)),'c-','LineWidth',lw) 
plot(F(:,4)./1000,P_F(:,4)./max(P_F(:,4)),'m-.','LineWidth',lw) 
plot(F(:,5)./1000,P_F(:,5)./max(P_F(:,5)),'g-','LineWidth',lw) 
plot(F(:,6)./1000,P_F(:,6)./max(P_F(:,6)),'k-.','LineWidth',lw) 
xlim([0 600])
%ylim([0 0.05])
xlabel('Frequency (kHz)')
ylabel('Normalized Amplitude')
legend('2mm','4mm','6mm','8mm','10mm','12mm','Location','northeast')
%% Ploting FFT in db
figure
lw = 2;
plot(F(:,1)./1000,db(P_F(:,1)./max(P_F(:,1))),'r-.','LineWidth',lw) 
hold on
plot(F(:,2)./1000,db(P_F(:,2)./max(P_F(:,2))),'b-.','LineWidth',lw) 
plot(F(:,3)./1000,db(P_F(:,3)./max(P_F(:,3))),'k-','LineWidth',lw) 
plot(F(:,4)./1000,db(P_F(:,4)./max(P_F(:,4))),'m-.','LineWidth',lw) 
plot(F(:,5)./1000,db(P_F(:,5)./max(P_F(:,5))),'g-.','LineWidth',lw) 
plot(F(:,6)./1000,db(P_F(:,6)./max(P_F(:,6))),'k-.','LineWidth',lw) 
xlim([0 600])
ylim([-80 0])
xlabel('Frequency (kHz)')
ylabel('Normalized Amplitude(db)')
legend('2mm','4mm','6mm','8mm','10mm','12mm','Location','northeast')
%% Normalized 2nd harmonic Amplitute with debonding size
D_size = [2 4 6 8 10 12];
for i=1:6
    amp_b=max(P_B(20:35,i));%./max(P_B(:,i));
    amp_f=max(P_F(20:35,i));%./max(P_F(:,i));
    Amp2_B(:,i)=amp_b;
    Amp2_F(:,i)=amp_f;
end
%% plot
figure
plot(D_size,Amp2_B,'r-*','LineWidth',2)
hold on
%plot(D_size,Amp2_B,'r-.','LineWidth',2)
plot(D_size,Amp2_F,'b-s','LineWidth',2)
%plot(D_size,Amp2_F,'b.-','LineWidth',2)
xlabel('Debonding Length (mm)')
ylabel('Amplitude')
legend('BackWard Scattering','Forward Scattering','Location','northwest')
%% 3rd harmonic Efficiency Calculation
D_size = [2 4 6 8 10 12];
for i=1:6
    amp_b=max(P_B(35:50,i));%./max(P_B(:,i));
    amp_f=max(P_F(35:50,i));%./max(P_F(:,i));
    Amp2_B(:,i)=amp_b;
    Amp2_F(:,i)=amp_f;
end
%% plot
figure
plot(D_size,Amp2_B,'r-d','LineWidth',2)
hold on
plot(D_size,Amp2_F,'b-s','LineWidth',2)
xlabel('Debonding Length (mm)')
ylabel('Amplitude')
legend('BackWard Scattering','Forward Scattering','Location','northeast')

