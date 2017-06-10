function [ vDelta, vTSimu ] = generator_Hawkes_ogataThin( N, mu, alpha, beta,  kernelType)
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: Generator of Hawkes Process with Power ('power') or Exponential ('exp') kernel
% Input: 
%      N: the number of events to be generated. #iet = n, #events = n+1
%      mu: average immigrating rate
%      alpha: branching factor
%      beta: decay exponent of memory kernel
%      kernelType:
%           If kernelType == 'exp',
%               \lambda = \mu + \sum_{t_i < t} (\alpha exp(-\beta(t - t_i)))
%           Else if kernelType == 'power',
%               \lambda = \mu + \sum_{t_i < t} (\alpha (t - t_i)^{-\beta}))
% Output:
%      vDelta: the vector of generated inter event time
%      vTSimu: the vector of generated event time from time 0

%
t = 0;
n = 0;
delta = 0;
vDelta = zeros(N,1);
vT = []; 
eps = 1e-10;
while n < N
    if strcmp(kernelType, 'exp')
         mt = mu + alpha * sum(exp(-beta*(t-vT)));
    elseif strcmp(kernelType, 'power')
         mt = mu + alpha * sum((t-vT+eps).^(-beta));       
    end
    s = randraw('exp', mt, [1 1] );
    u = rand;
    t = t + s;
    if strcmp(kernelType, 'exp')
         lambdaStarTplusS = mu + alpha * sum(exp(-beta*(t-vT)));
    elseif strcmp(kernelType, 'power')
         lambdaStarTplusS = mu + alpha * sum((t-vT+eps).^(-beta));       
    end
    if u * mt <= lambdaStarTplusS        
        n = n+1;
        vDelta(n) = s;
        vT = [vT; t];
    end
    
end

vTSimu = [0; cumsum(vDelta)];

end

