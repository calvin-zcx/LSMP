function [ vDelta, vTSimu ] = generator_PoissonProcess( n, mu )
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: Generator of Poisson Process 
% Input: 
%      n: the number of events to be generated. #iet = n, #events = n+1
%      mu: with average inter event time mu
% Output:
%      vDelta: the vector of generated inter event time
%      vTSimu: the vector of generated event time from time 0
% Algorithm:
%      f(x) = lambda * exp{-lambda * x}
%           = 1/mu * exp{-1/mu * x}
%      i Generate U ~ unif(0, 1).
%       ii Set X = - 1/lambda * Ln(U)
%                = - mu * Ln(U);

vDelta = zeros(n,1);
for i = 1:n
   deltat = -mu * log(rand);
   vDelta(i) = deltat;
end

vTSimu = [0; cumsum(vDelta)];
end

