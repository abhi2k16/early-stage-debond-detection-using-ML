%%
clear all; clear;
x_fem=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\UnDamPZT\U1S0PDL140kHzSEN150mmUnDamPlaneStiff.rpt');
x_dam_fem=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam10mm\U1S0PDL140kHzSEN150mmDam10mmPlaneStiff.rpt');
%%
%x_exp2(2:752,3)=x_exp2(2:752,3).*0.1;
figure
plot(x_fem.data(:,1).*1e6,x_fem.data(:,2)./max(x_fem.data(:,2)),'k-','LineWidth',1)
hold on
plot(x_dam_fem.data(:,1).*1e6,x_dam_fem.data(:,2)./max(x_dam_fem.data(:,2)),'k-.')
xlim([0 300])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
%%
%%
x3=x_fem.data(:,2);
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
figure
plot(f,P1,'b-','LineWidth',1) 
%title('FFT for sensor at 330mm')
xlabel('f(Hz)')
ylabel('Amplitute')
%%