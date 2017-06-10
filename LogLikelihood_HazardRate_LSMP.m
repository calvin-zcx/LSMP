function [ LogLikeli, gradients ] = LogLikelihood_HazardRate_LSMP(X, vT, memLength)
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: Calculate the log-likelihood function of LSMP process, 
%       and the gradients with respect to the parameters of LSMP hazard rate.
% Input: 
%      X: model parameters of the hazard rate of LSMP (Equation 2 in the paper)
%      vT: a vector of event time, assuming the first event happens at time 0.
%      memLength: memory length

% Output:
%      LogLikeli: the value of the loglikelihood function. A real value.
%      gradients: a vector of each model parameter.
%                 = -[gradientOfLambda0; gradientOft0; gradientOftheta; gradientOflambdaP; gradientOfa; gradientOfT];    

lambda0 = X(1);
t0 =  X(2);
theta = X(3);
lambdaP = X(4);
a = X(5);
T =  X(6);
 
tn = vT(end);
N = length(vT);
A = zeros(N,1);
B = zeros(N,1);
C = zeros(N,1);

if theta == 1
    theta = 1.00000001;
end

%% Rate part
for i = 2:N 
    ti = vT(i);
    vMemT = vT(max(1,i-memLength):i-1);
    A(i) = sum( ((ti-vMemT)./t0 + 1).^(-theta) );
    B(i) = sum( (theta/t0^2).*((ti-vMemT)).*((ti-vMemT)./t0 + 1).^(-theta-1) );
    C(i) = sum( ((ti-vMemT)./t0 +1).^(-theta) .* log((ti-vMemT)./t0 + 1) );
end
valLogRate = sum( log( lambdaP*a*(vT./T+1).^(a-1)  +  lambda0*A ) );

%% Integral part
vTm = zeros(size(vT));
left = max(N - memLength + 1, 1);
vTm(1:left-1) = vT(1+memLength:end);
vTm(left:end) = vT(end);
vInteg = lambda0*t0/(1-theta)*( ((vTm - vT)./t0 + 1).^(1-theta) - 1 );
valIntegral = sum(vInteg);

valInteralTrend = lambdaP*T*((tn/T+1)^a - 1);
%% -LogLikelihood for minimize function.
LogLikeli = -(-valInteralTrend - valIntegral + valLogRate) ;
 
%% Gradients
if nargout > 1 % gradient required
    %% auxiliary function
    D = lambdaP*a*((vT./T+1).^(a-1)) + lambda0.*A;
    
    %% gradientOfLambda0    
    gradientOfLambda0 = -valIntegral./lambda0 + sum( A./D );
    
    %% gradientOft0
%     gradientOft0 = lambda0*N/(1-theta) -  sum( ...
%      lambda0/(1-theta)*( (vTm - vT)./t0 + 1 ).^(-theta) .* (1+theta/t0.*(vTm-vT)) ...
%      ) + sum( lambda0.*B./D  );
    gradientOft0 = lambda0*N/(1-theta) -  lambda0/(1-theta)*sum( ...
     ( (vTm - vT)./t0 + 1 ).^(-theta) .* (1+theta/t0.*(vTm-vT)) ...
     ) + sum( lambda0.*B./D  );
% %     gradientOft0 = -valIntegral/t0 + lambda0/t0*sum(((vTm-vT)./t0+1).^(-theta).*(vTm-vT)) + sum( lambda0.*B./D  );
    %% gradientOftheta
    gradientOftheta = -valIntegral/(1-theta) + sum(...
        lambda0*t0/(1-theta)*((vTm - vT)./t0 + 1).^(1-theta).*log((vTm - vT)./t0 + 1)...
        ) - sum( lambda0*C./D  );  
    %% gradientOflambdaP
    gradientOflambdaP = -T*((tn/T+1)^a-1) + sum( a*(vT./T+1).^(a-1)./D );
%     
%     %% gradientOfa
    gradientOfa = -lambdaP*T*(tn/T+1)^a*log(tn/T+1) + sum(...
        lambdaP*(vT./T+1).^(a-1).*(1 + a*log(vT./T+1))./D ...
        );
%     
    %% gradientOfT
    %% Due to the scale of T^2, I seperate it into T*T in different part.
    gradientOfT = -lambdaP*((tn/T+1)^a-1) + lambdaP*a*tn/T*(tn/T+1)^(a-1) + ...
        -lambdaP*a*(a-1)/(T^2)*sum(...
        (vT.*(vT./T+1).^(a-2))./D ...
        );
 gradients = -[gradientOfLambda0; gradientOft0; gradientOftheta; gradientOflambdaP; gradientOfa; gradientOfT];    
   

end

end


