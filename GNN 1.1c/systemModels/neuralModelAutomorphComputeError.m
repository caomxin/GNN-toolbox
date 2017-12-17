function [e,outState]=neuralModelAutomorphComputeError(dataset,x,optimalParam)

global dataSet dynamicSystem learning

%% x will be empty except when called to test the results
if isempty(x) && strcmp(dataset,'trainSet')
    in=[dynamicSystem.state;dataSet.trainSet.nodeLabels];
elseif isempty(x) && strcmp(dataset,'validationSet')
    in=[learning.current.validationState;dataSet.validationSet.nodeLabels];
else
    in=[x;dataSet.(dataset).nodeLabels];
end


%[outState.out,outState.outNetState]=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet');
outState.outNetState=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet',optimalParam);

sz=size(outState.outNetState.outs,2);
if size(unique(outState.outNetState.outs),2)==sz
    disp('resolved')
end

%% Compute the error.
eps=0.001;
e=0;
outState.delta=zeros(1,sz);
% outState.outNetState.outs
for n=1:sz
    for k=n+1:sz
        d1=outState.outNetState.outs(n)-outState.outNetState.outs(k);
        term=1/(d1*d1+eps);
        e=e+term;
        d2=term*term*d1;
        outState.delta(n)=outState.delta(n)+d2;
        outState.delta(k)=outState.delta(k)-d2;
    end
end
e=e*0.5;
outState.delta=-outState.delta;
% e
% outState.delta








%% Compute the error. The error is the quadratic difference of the targets from current outputs
%% In general  supervision may be placed only on some outputs: matrix "maksMatrix" allows to select the supervised outputs
%outState.delta=(dataSet.(dataset).maskMatrix*outState.outNetState.outs')';
%e= outState.outNetState.outs*dataSet.(dataset).maskMatrix*outState.outNetState.outs'/2;
