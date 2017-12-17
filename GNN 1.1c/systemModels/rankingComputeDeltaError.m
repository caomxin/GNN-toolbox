%function [gradient,dInputs]=rankingDeltaError(x,p,outState,sys)
function [gradient,dInputs]=rankingDeltaError(outState)

global dynamicSystem learning

% sx=size(x);
sx=size(dynamicSystem.state,1);

if isempty(outState)
    % [gradient.outNet,dInputs]=feval(sys.outNet.backwardFunction,p.outNet,outState.outNetState,outState.delta);
    % dInputs=dInputs(1:sx(1),:);
    [gradient.outNet,dInputs]=feval(dynamicSystem.config.outNet.backwardFunction,dynamicSystem.parameters.outNet,...
        learning.current.outState.outNetState,learning.current.outState.delta);
else
    % [gradient.outNet,dInputs]=feval(sys.outNet.backwardFunction,p.outNet,outState.outNetState,outState.delta);
    % dInputs=dInputs(1:sx(1),:);
    [gradient.outNet,dInputs]=feval(dynamicSystem.config.outNet.backwardFunction,dynamicSystem.parameters.outNet,...
        outState.outNetState,outState.delta);
end

dInputs=dInputs(1:sx,:);
