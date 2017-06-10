function [ vDelta, vTSimu ] = generator_LSMP_inverseMethod( N, lambda0, t0, theta, lambdaP, memLength, a, T, initialvT)
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: Generator of LSMP process by inverse method. (Algorithm 1 in the paper)
% Input: 
%      N: the number of events to be generated
%      lambda0: the short term event rate
%      t0: the short-term time scale
%      theta: the short-term fizzling exponent
%      lambdaP: the long-term event rate
%      memLength: memory length
%      a: the long-term growth exponent
%      T: the long-term time scale
%      X: model parameters of the hazard rate of LSMP (Equation 2 in the paper)
%      initialvT: a vector of first length(initialvT) given event time, assuming the first event happens at time 0.

% Output:
%      vDelta: the vector of generated inter event time
%      vTSimu: the vector of generated event time from time 0


if isempty(initialvT)
    n = 1;
    u = rand;
    t1 = T*((-log(u)/(lambdaP*T)+1)^(1/a)-1);
    vT = [t1];
    vDelta(n) = t1;
    t = t1;
else
    n = length(initialvT);
    vT = initialvT;
    vDelta = diff(initialvT);
    t = vT(end);
end

n = n + 1;
while n < N
    %s = randraw('exp', 1, [1 1] );  % = -log(rand)
   u = rand; %generate a random variable from uniform distribution U([0,1]) 
   tolerance = 1e-8; %precision of error
   nmax = 2000;  % maximum number of iteration of Newton's iterative method
   x0 =  t ;%- log(u)/(lambdaP*a*(t/T+1)^(a-1));
   [ x, ex ] = mynewton(tolerance, nmax,  x0,  u, vT, lambda0, t0, theta, lambdaP, memLength, a, T );
    
    tAnswer = x(end);  
    n = n+1; 
    vDelta = [vDelta; tAnswer-t];
    t =  tAnswer;
    vT = [vT; tAnswer];
end

vTSimu = [0; cumsum(vDelta)];
end

function [ value, grad ] = integralOfHazardRate(t, offset, vT, lambda0, t0, theta, lambdaP, memLength, a, T)
% offset = Paras(1);
% vT = Paras(2);
% lambda0 = Paras(3);
% t0 = Paras(4);
% theta = Paras(5);
% lambdaP = Paras(6);
% memLength = Paras(7);
% a = Paras(8);
% T = Paras(9); 
if isempty(vT)
    vT = 0;
    tn = 0;
else
    tn = vT(end);
end

N = length(vT);
left = max(N - memLength + 1, 1);
vTm = vT(left:end);


value = lambda0*t0/(1-theta) * sum(...
    ( ((t - vTm)./t0 + 1).^(1-theta) - ((tn - vTm)./t0 + 1).^(1-theta) )...
    ) ...
    + lambdaP*T*((t/T+1)^a - (tn/T+1)^a) ...
    +log(offset);

if nargout > 1
    grad = lambdaP*a*(t/T+1)^(a-1) + lambda0*sum( 1./((t-vTm)./t0+1).^theta );
end

end

function [ x, ex ] = mynewton(tol, nmax,  x0,  offset, vT, lambda0, t0, theta, lambdaP, memLength, a, T )
% Algorithm 2 in the paper
   [ f, grad ] = integralOfHazardRate(x0, offset, vT, lambda0, t0, theta, lambdaP, memLength, a, T);
    x(1) = x0 - (f/grad);
    ex(1) = abs(f);
    k = 2;
    while (ex(k-1) >= tol) && (k <= nmax)
        [ f, grad ] = integralOfHazardRate(x(k-1), offset, vT, lambda0, t0, theta, lambdaP, memLength, a, T);
        x(k) = x(k-1) - (f/grad);
        ex(k) = abs(f);%abs(x(k)-x(k-1));
        k = k+1;
    end
    
    if length(ex) < nmax
        %fprintf('get answer\n');
    else
        fprintf('No answer found!\n');
    end
end