%%
clear; clear all;
DispCurve=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\dispersion_curve_AL_150mm.xlsx');
ExpDispCurve=xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Exp_group_velocity.xlsx');
% freq(MHz) gr_speed (m/msec) :colmn-1 & 2 for S0 freq & gr_speed and colmn-11 & 12 for A0 freq &  gr_speed
%% Plotting of S0 and A0 group speed 
figure 
plot(DispCurve(8:80,1).*1000,DispCurve(8:80,2).*1000,'k-.','LineWidth',2)
xlim([0 610])
ylim([0 6000])
hold on
plot(ExpDispCurve(:,1),ExpDispCurve(:,2),'b*','LineWidth',2)
plot(DispCurve(462:571,11).*1000,DispCurve(462:571,12).*1000,'k-.','LineWidth',2)
plot(ExpDispCurve(:,1),ExpDispCurve(:,3),'b*','LineWidth',2)
xlabel('Frequency (kHz)')
ylabel('Group Vel. (m/sec)')
legend('Analytical','Experimental')
%%