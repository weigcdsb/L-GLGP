function logp = normpdfln(x,mu,Sigma,iSigma)

x = x(:);
if nargin<4
    q = size(Sigma,1);
    iSigma = Sigma \ eye(q);
    
    logdet_iSigma = -2*sum(log(diag(chol(Sigma))));
else
    logdet_iSigma = 2*sum(log(diag(chol(iSigma))));
end

logp = 0.5*logdet_iSigma - 0.5*(x-mu)'*(iSigma*(x-mu));

return;