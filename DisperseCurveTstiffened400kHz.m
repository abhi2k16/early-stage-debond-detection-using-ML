%%
clear; clear all;
epx_grp_phs_vel = xlsread('C:\Users\abhij\Desktop\WaveModel\WaveModel_EXP\ExperimentalGrpPhsVelocityData.xlsx');
epx_grp_phs_vel_2 = xlsread('C:\Users\abhij\Desktop\WaveModel\WaveModel_EXP\ExperimentalGrpPhsVelocityData_2.xlsx');
exp_dispersion = xlsread('C:\Users\abhij\Desktop\WaveModel\WaveModel_EXP\ExperimentalDispersionCurve.xlsx');
exp_dispersion_2= xlsread('C:\Users\abhij\Desktop\WaveModel\WaveModel_EXP\ExperimentalDispersionCurve_2.xlsx');
PZT_exp_disp = xlsread('C:\Users\abhij\Desktop\WaveModel\PZT_Model\Exp_group_velocity.xlsx');
load('DisperseCurveT-stiffened400kHz.mat')
%%
freq_thr =Frequency_Hz(1:401)./1000; % In kHz
freq_exp =epx_grp_phs_vel(:,1);      % In kHz 
freq_exp_2 =epx_grp_phs_vel_2(:,2); 
PZT_exp_freq = PZT_exp_disp(:,1);
PZT_exp_GpVl_S0 = PZT_exp_disp(:,2);
PZT_exp_GpVl_A0 = PZT_exp_disp(:,3);
exp_grp_vel_A0 = epx_grp_phs_vel(:,6);  % Grp Velocity A0
exp_grp_vel_SH0 = epx_grp_phs_vel_2(:,9);  % Grp Velocity SH0
exp_grp_vel_S0 = epx_grp_phs_vel_2(:,11);  % Grp Velocity S0
exp_grp_vel_F0 =[2105 2285 2330 2320 2330 2340];
%% Group Velocity
GV=zeros(5,400);
for i=2:401
    GV(:,i) = sort(Group_Velocity_m_s(:,i),1);
end
%Phase Velocity
PhV = zeros(5,400);
for i=2:401
    PhV(:,i) = sort(Phase_Velocity_m_s(:,i),1);
end
%% Plotting Group Velocity
figure; 
c1=2; c2=350;
%plot(freq_exp_2,exp_grp_vel_S0(:,:),'r*-','LineWidth',1)
plot(PZT_exp_freq,PZT_exp_GpVl_S0*0.95,'r*','LineWidth',1)
hold on
%plot(freq_exp(5:20),exp_dispersion_2(5:20,6)*0.93,'bd-','LineWidth',1)
plot(freq_thr(c1:c2),GV(1,c1:c2)*1.5,'k.','LineWidth',4)
%plot(freq_exp_2,exp_grp_vel_F0(:,:)*0.95,'r*-','LineWidth',1)
plot(PZT_exp_freq,PZT_exp_GpVl_A0*0.9,'r*','LineWidth',1)
plot(freq_thr(c1:c2),GV(2,c1:c2)*1.5,'k.','LineWidth',4)
plot(freq_thr(c1:c2),GV(3,c1:c2)*1.5,'k.','LineWidth',4)
plot(freq_thr(c1:c2),GV(4,c1:c2)*1.2,'k.','LineWidth',4)
plot(freq_thr(c1:c2),GV(5,c1:c2),'k.','LineWidth',4)
%plot(freq_exp_2,exp_grp_vel_SH0,'md-','LineWidth',2)
xlabel('Frequency (kHz)')
ylabel('Group Velocity (m/sec)')
legend({'Experimental','SAFE'},'Location','northeast')
%% Plotting Phase Velocity
figure; 
plot(PhV(1,2:401)*1.5,'r.')
hold on
plot(PhV(2,2:401)*1.5,'g.')
plot(PhV(3,2:401)*1.5,'b.')
plot(PhV(4,2:401)*1.5,'k.')
plot(PhV(5,2:401),'m.')
%% Plotting Flexural phase velocity
figure; 
plot(301:1:311,PhV(3,301:311)*1.5,'k.')
hold on
plot(12:1:300,PhV(4,12:300)*1.5,'k.')
plot(312:1:400,PhV(4,312:400)*1.5,'k.')
%% calculating Wave length
Phase_vel=[PhV(4,12:300)*1.5 PhV(3,301:311)*1.5 PhV(4,312:400)*1.5];
WAVELENGTH =[Wavelength_m(4,12:300)*1.5 Wavelength_m(3,301:311)*1.5 Wavelength_m(4,312:400)*1.5];
for i=12:1:400
    wavelength = Phase_vel./(i*1000);
end
%%
figure
plot(WAVELENGTH,'k.')
%% Plotting Wavelength
figure; 
plot(20:401,Wavelength_m(1,20:401)*1,'r.')
hold on
plot(20:401,Wavelength_m(2,20:401)*1,'g.')
plot(20:401,Wavelength_m(3,20:401)*1,'b.')
plot(20:401,Wavelength_m(4,20:401)*1,'k.')
plot(20:401,Wavelength_m(5,20:401),'m.')
%% Plotting Flexural mode Wave lenght
figure; 
n=1.5;
plot(20:1:87,Wavelength_m(3,20:87)*n,'g.','LineWidth',7)
hold on
plot(89:1:103,Wavelength_m(2,89:103)*n,'g.','LineWidth',7)
plot(104:107,Wavelength_m(1,104:107)*n,'g.','LineWidth',7)
plot(109:1:115,Wavelength_m(2,109:115)*n,'g.','LineWidth',7)
plot(116:136,Wavelength_m(1,116:136)*n,'g.','LineWidth',7)
plot(138:1:144,Wavelength_m(2,138:144)*n,'g.','LineWidth',7)
plot(146:1:168,Wavelength_m(2,146:168)*n,'g.','LineWidth',7)
plot(170:1:173,Wavelength_m(4,170:173)*n,'g.','LineWidth',7)
plot(175:1:309,Wavelength_m(3,175:309)*n,'g.','LineWidth',7)
plot(311:1:350,Wavelength_m(4,311:350)*n,'g.','LineWidth',7)
xlabel('Frequency (kHz)')
ylabel('Wavelength (m)')
xlim([50 350])
ylim([0 0.025])
%%