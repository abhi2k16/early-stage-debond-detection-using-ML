%%
clear all; clear;
x_fem_Plane=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\UnDamPZT\U1S0PDL140kHzSEN150mmUnDamPlaneStiff.rpt');
x_fem_Plate=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\UnDamPZT\U1S0PDL140kHzSEN150mmUnDamPlane.rpt');
x_fem_T=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\UnDamPZT\U1S0PDL140kHzSEN150mmUnDam.rpt');
x_dam_Plane=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam10mm\U1S0PDL140kHzSEN150mmDam10mmPlaneStiff.rpt');
x_dam_T=importdata('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Dam10mm\U1S0PDL140kHzSEN150mmDam10mm.rpt');
%%
figure
x_fem_Plate.data(1:5370,2)=x_fem_Plate.data(1:5370,2).*-1
plot(x_fem_Plane.data(:,1).*1e6,x_fem_Plane.data(:,2)./max(x_fem_Plane.data(:,2)),'r-','LineWidth',1.5)
hold on
plot(x_fem_T.data(:,1).*1e6,x_fem_T.data(:,2)./max(x_fem_T.data(:,2)),'b-','LineWidth',1.5)
plot(x_fem_Plate.data(:,1).*1e6,x_fem_Plate.data(:,2)./max(x_fem_Plate.data(:,2)),'g-','LineWidth',1.5)
xlim([0 100])
ylim([-1 1])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('Plane-stiffener','T-stiffener','Plate')
%% Hilbert Transform plot
figure
plot(x_fem_Plane.data(:,1).*1e6,hilbert(x_fem_Plane.data(:,2))./max(abs(hilbert(x_fem_Plane.data(:,2)))),'r-','LineWidth',1.5)
hold on
plot(x_fem_T.data(:,1).*1e6,hilbert(x_fem_T.data(:,2))./max(abs(hilbert(x_fem_T.data(:,2)))),'b-','LineWidth',1.5)
plot(x_fem_Plate.data(:,1).*1e6,hilbert(x_fem_Plate.data(:,2))./max(abs(hilbert(x_fem_Plate.data(:,2)))),'g-','LineWidth',1.5)
xlim([0 100])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('Plane-stiffener','T-stiffener','Plate')
%% Subplot
figure
subplot(2,1,1);
plot(x_fem_Plane.data(:,1).*1e6,x_fem_Plane.data(:,2)./max(x_fem_Plane.data(:,2)),'r-','LineWidth',1.5)
xlim([0 100])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('T-stiffener')
subplot(2,1,2); 
plot(x_fem_T.data(:,1).*1e6,x_fem_T.data(:,2)./max(x_fem_T.data(:,2)),'b-','LineWidth',1.5)
xlim([0 100])
xlabel('Time (µ sec)')
ylabel('Normalized Amplitude')
legend('Plane-stiffener')
%%
figure
for i=1:2
    if i==1
        x3=x_dam_Plane.data(1:18012,2);
    else 
        x3=x_dam_T.data(1:18012,3);
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
    xlim([0 900])
    ylim([0 1])
end
 xlabel('Frequency kHz')
 ylabel('Normalized Amplitute')
 legend('Plane-stiffener','T-stiffener')
%%