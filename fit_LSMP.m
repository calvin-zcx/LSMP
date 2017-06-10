function [ memLength, paras, fval, vMem, vval  ] = fit_LSMP( vT )
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: To learn the parameters of the Long Short Memory Process.
% Input: 
%      vT: a vector of event time, assuming the first event happens at time 0.
% Output:
%      memLength: (Model parameter) learned memory length
%      paras: (Model parameter) [lambda0, t0, theta, lambdaP, a, T]
%      
%      fval: (For evaluation) Objective function value at the solution, returned as a real number. 
%      vMem: (For evaluation) the vector of iterated memory length
%      vval: (For evaluation) the objetive function value at corresponding memory length.
   

vMem = 0:2:10; %0:10; % the vector of iterated memory length   
%1;%10;%1:10;%[20, 40, 60];% [1 5 10 15 20 30 40 50];
mN = length(vMem); % the number of trials of memory length 

vval = zeros(mN,1); %Objective function value holder.
vpara = cell(mN,1); %parameters holder. 

iat = (diff(vT)); %the vector of inter event time
miat = mean(iat); %average value of inter event time

lastval = 0;
mNBreak = mN;

for i = 1: mN
%% Set learning algorithm
    %     options = optimoptions('fmincon','Display','iter','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true);
    %     options = optimoptions('fmincon','Display','iter','CheckGradients', true,'Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true); 
    options = optimoptions('fmincon','Display','none',...
        'Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true);
%% Set initial values of the parameters.    
    memLength = vMem(i);    
    lambda0 = rand;  %.1;0.01;
    t0 =  rand*10;   %1;
    theta =  3*rand; %1.5*rand+1.5; %2; %1.5;
    
    lambdaP = 1./miat;
    a =   rand*5;        %1;rand*2
    TInf =   1./lambdaP; % * 100; %3600*24*2;
    
%     lambda0, t0, theta, lambdaP, a, TInf   
    lb = [0,    0,    0,     0,     0,    0];
    ub = [Inf,  Inf,  Inf,   Inf,   Inf,  Inf];

%% Minimize the -LogLikelihood of LSMP.
% [Paras,fval,EXITFLAG,OUTPUT] = 
    [Paras,fval] = fmincon(@(X) LogLikelihood_HazardRate_LSMP(X, vT, memLength) , ...
               [lambda0, t0, theta, lambdaP, a, TInf], [],[],[],[], lb,ub, [], options); 
%% Collect learned values at memory length memLength.
     vval(i) = fval;    
     vpara{i}  = Paras;
     
%      if abs(fval - lastval) < 1
%          mNBreak = i;
%          break;
%      end
%      lambda0_learned = Paras(1);
%      t0_learned = Paras(2);
%      theta_learned =Paras(3);
%      lambdaP_learned = Paras(4);
%      a_learned = Paras(5);
%      TInf = Paras(6);
end  

% [Y, I] = min(vval(1:mNBreak));
[Y, I] = min(vval);
fval = Y;
memLength =  vMem(I);
paras = vpara{I};

end

