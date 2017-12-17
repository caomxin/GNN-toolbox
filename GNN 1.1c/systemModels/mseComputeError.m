function [e,outState]=neuralModelComputeError(dataset,x,optimalParam)

global dataSet dynamicSystem learning

%% x will be empty except when called to test the results
if isempty(x) && strcmp(dataset,'trainSet')
    in=[dynamicSystem.state;dataSet.trainSet.nodeLabels];
elseif isempty(x) && strcmp(dataset,'validationSet')
    in=[learning.current.validationState;dataSet.validationSet.nodeLabels];
else
    in=[x;dataSet.(dataset).nodeLabels];
end
    
%[outState.out,outState.outNetState]=feval(sys.outNet.forwardFunction,in,p.outNet);
%[outState.out,outState.outNetState]=feval(sys.outNet.forwardFunction,in,'outNet');
outState.outNetState=feval(dynamicSystem.config.outNet.forwardFunction,in,'outNet',optimalParam);

%% Compute the error. The error is the quadratic difference of the targets from current outputs
%% In general  supervision may be placed only on some outputs: matrix
%% "maksMatrix" allows to select the supervised outputs
outState.delta=(outState.outNetState.outs-dataSet.(dataset).targets)*dataSet.(dataset).maskMatrix; % sarebbe maskMatrix' ma e' simmetrica

if size(outState.delta,1)>1
    % multiclass problem: the error is summed over all the components
    e=0;
    for (i=1:size(outState.delta,1))
        e = e+outState.delta(i,:)*outState.delta(i,:)'/2;
    end
else
    e=outState.delta *outState.delta'/2;
end


