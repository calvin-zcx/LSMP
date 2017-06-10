function [ LogLikeli, gradients ] = LogLikelihood_HazardRate_Hawkes(X, vT, kernelType)
% Version 1.0
% Data: 2017/06/09 
% Author: Chengxi Zang
% Venue: KDD 2017, Long Short Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity

% Goal: Calculate the log-likelihood function of Hawkes process, 
%       and the gradients with respect to the parameters of LSMP hazard rate.
% Input: 
%      X: model parameters of the hazard rate of LSMP, [mu, alpha, beta] 
%      vT: a vector of event time, assuming the first event happens at time 0.
%      kernelType:
%           If kernelType == 'exp',
%               \lambda = \mu + \sum_{t_i < t} (\alpha exp(-\beta(t - t_i)))
%           Else if kernelType == 'power',
%               \lambda = \mu + \sum_{t_i < t} (\alpha (t - t_i)^{-\beta}))

% Output:
%      LogLikeli: the value of the loglikelihood function. A real value.
%      gradients: a vector of each model parameter.
%                 = -[gradientOfmu; gradientOfalpha; gradientOfbeta];

mu = X(1);
alpha =  X(2);
beta = X(3);
 
 
tn = vT(end);
N = length(vT);
A = zeros(N,1);
B = zeros(N,1);
 

if strcmp(kernelType, 'power') && beta == 1
    beta = 1.00000001;
end

eps = 1e-8;
%% Rate part
for i = 2:N 
    ti = vT(i);
    vMemT = vT(1:i-1);
    if strcmp(kernelType, 'exp')
        A(i) = sum (exp(-beta.*(ti-vMemT)) );
        B(i) = sum (exp(-beta.*(ti-vMemT)) .*(ti-vMemT) );
    elseif strcmp(kernelType, 'power')
        A(i) =  sum( ((ti-vMemT+eps)).^(-beta) ); 
        B(i) = sum((ti-vMemT+eps).^(-beta).*log(ti-vMemT+eps));
    end  
end

valLogRate = sum( log( mu + alpha*A ) );

%% Integral part
if strcmp(kernelType, 'exp')
    vInteg = -alpha/beta*(exp(-beta*(tn-vT))-1);
elseif strcmp(kernelType, 'power')
    vInteg = alpha/(1-beta)*(tn-vT+eps).^(1-beta);
end
valIntegral = sum(vInteg);
valInteralTrend = mu*tn;
%% -LogLikelihood for minimize function.
LogLikeli = -(-valInteralTrend - valIntegral + valLogRate) ;
 
%% Gradients
if nargout > 1 % gradient required
    %% auxiliary function
    D = mu + alpha.*A;
    
    %% gradientOfmu   
    gradientOfmu = -tn + sum(1./D);
    
    if strcmp(kernelType, 'exp')
        %% gradientOfalpha
        gradientOfalpha = 1/beta*sum(exp(-(tn-vT))-1) + sum(A./D);
        %% gradientOfbeta
        gradientOfbeta = -alpha*sum(...
            1./beta*(tn-vT).*exp(-beta*(tn-vT)) + 1./beta^2*(exp(-beta*(tn-vT))-1) ...
            )... 
            - sum(alpha*B./D);
     elseif strcmp(kernelType, 'power')
         %% gradientOfalpha
         gradientOfalpha = -1./(1-beta)*sum((tn-vT+eps).^(1-beta)) + sum(A./D);
         %% gradientOfbeta
         gradientOfbeta = -alpha*sum(...
             1./(1-beta)^2.*(tn-vT+eps).^(1-beta) - 1./(1-beta)*(tn-vT+eps).^(1-beta).*log(tn-vT+eps)...
             )...
             -sum(alpha*B./D);
    end
 gradients = -[gradientOfmu; gradientOfalpha; gradientOfbeta];

end

end


