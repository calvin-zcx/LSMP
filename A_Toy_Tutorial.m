% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity
% Goal: A toy tutorial to showcase how to fit, 
%       generate and plot the growth dynamics by different process.
% Notice: Run multiple times for different simulation results.
clear
close all

% Load Empirical Data
DATA = load('test_data');
vT = DATA.vT;
iat = diff(vT);
n = length(vT);
maxT = vT(end);

%% Fit Long Short Memory Process, and generate simulated process by LSMP 
[ memLength, Paras, fval, vMem, vval ] =   fit_LSMP( vT );

lambda0_learned = Paras(1);
t0_learned = Paras(2);
theta_learned =Paras(3);
lambdaP_learned = Paras(4);
a_learned = Paras(5);
TInf = Paras(6);
[iatSimuLSMP, vTSimuLSMP] = generator_LSMP_inverseMethod( n, lambda0_learned, t0_learned, theta_learned, lambdaP_learned, memLength, a_learned,  TInf, []);

%% Fit Hawkes Process with exponential and power law kernel. Then simulate.
kernelType = 'exp';
[parasHawkes, fvalHawkes ] = fit_Hawkes( vT, kernelType );  
mu = parasHawkes(1); alpha = parasHawkes(2); beta = parasHawkes(3);
[iatSimuHawkesE,  vTSimuHawkesE] = generator_Hawkes_ogataThin( n-1, mu, alpha, beta,  kernelType);


kernelType = 'power';
[parasHawkes, fvalHawkes ] = fit_Hawkes( vT, kernelType );  
mu = parasHawkes(1); alpha = parasHawkes(2); beta = parasHawkes(3);
[iatSimuHawkesP,  vTSimuHawkesP] = generator_Hawkes_ogataThin( n-1, mu, alpha, beta,  kernelType);

%% Fit Poisson Process and Self-feeding process. Then simulate.
mu = mean(iat);
[ vDeltaPP, vTSimuPP ] = generator_PoissonProcess( n-1, mu );     
[ vDeltaSFP, vTSimuSFP ] = generator_SelfFeedingProcess( n-1, mu, 1 );   
endsf = find(vTSimuSFP>maxT, 1, 'first');

%% Plot the result.
h = figure;
dd = 10; %% Set large sample gap, say 20, for less dots
h1 = plot(vT(1:dd:end), 1:dd: n, '-o',...
          vTSimuLSMP(1:dd:end), 1:dd:n, '-s',...
          vTSimuPP(1:dd:end), 1:dd:n, ':',...
          vTSimuHawkesE(1:dd:end), 1:dd:n, '--', ...
          vTSimuHawkesP(1:dd:end), 1:dd:n, '-.', ...
          vTSimuSFP(1:dd:endsf), 1:dd:endsf, '-',...
    'MarkerSize', 9, 'LineWidth',1.2); 
legend1 = legend({'Real', 'LSmP', 'PP', 'HWK-E', 'HWK-P', 'SFP'});

set(legend1,...
    'Position',[0.175 0.523 0.183 0.363],...
    'FontSize',14);
grid
set(gca,'FontSize',15);
xlabel('Time (s)','fontsize', 17);
ylabel('Number of Friends', 'fontsize', 17);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
