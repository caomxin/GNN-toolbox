% function [gradient,dInputs]=computeDeltaError(x,p,outState,sys)
function [gradient,dInputs]=neuralModelQuadraticComputeDeltaError(outState)

global dynamicSystem learning

% sx=size(x);
sx=size(dynamicSystem.state,1);

% [gradient.outNet,dInputs]=feval(sys.outNet.backwardFunction,p.outNet,outState.outNetState,outState.delta);
[gradient.outNet,dInputs]=feval(dynamicSystem.config.outNet.backwardFunction,dynamicSystem.parameters.outNet,...
    learning.current.outState.outNetState,learning.current.outState.delta);

% dInputs=dInputs(1:sx(1),:);
dInputs=dInputs(1:sx,:);