function [ paras, fval ] = fit_Hawkes( vT, kernelType )
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: To learn the parameters of the Hawkes Process.
% Input: 
%      vT: a vector of event time, assuming the first event happens at time 0.
%      kernelType:
%           If kernelType == 'exp',
%               \lambda = \mu + \sum_{t_i < t} (\alpha exp(-\beta(t - t_i)))
%           Else if kernelType == 'power',
%               \lambda = \mu + \sum_{t_i < t} (\alpha (t - t_i)^{-\beta}))
% 
% Output:
%      paras: (Model parameter) [mu, alpha, beta]    
%      fval: (For evaluation) Objective function value at the solution, returned as a real number. 

%     options = optimoptions('fmincon','Display','iter','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true);
%     options = optimoptions('fmincon','Display','iter','CheckGradients', true,'Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true);
options = optimoptions('fmincon','Display','none','Algorithm','trust-region-reflective','SpecifyObjectiveGradient',true);
mu = rand;
alpha = rand;
beta = rand;
 
lb = [0, 0,     0 ];
ub = [Inf, Inf, Inf ];

% [Paras,fval,EXITFLAG,OUTPUT] 
[paras,fval] = fmincon(@(X) LogLikelihood_HazardRate_Hawkes(X, vT, kernelType) , ...
               [mu, alpha, beta], [],[],[],[], lb,ub, [], options); 
     
end

