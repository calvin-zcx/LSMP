function [ vDelta, vTSimu ] = generator_SelfFeedingProcess( n, mu, rho )
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: Generator of Self Feeding Process 
% Input: 
%      n: the number of events to be generated. #iet = n, #events = n+1
%      mu: with average inter event time mu
%      mu: scaling exponent of mu
% Output:
%      vDelta: the vector of generated inter event time
%      vTSimu: the vector of generated event time from time 0
% Algorithm: SFP (n, mu, rho)
%      d(1) = mu
%      d(t) = ExpoDistri{mean: d(t-1) + mu^rho/e}
%      delta_t = d(t) ^(1/rho) 
%
deltat = mu;
vDelta = zeros(n,1);
for i = 1:n
   deltat = -(deltat + mu^rho/exp(1));%*(exp(i./50)./(1+exp(i./50))));%*(1+0.0001)^i
   deltat = deltat * log(rand);
   vDelta(i) = deltat^(1/rho);
end

vTSimu = [0; cumsum(vDelta)];
end

