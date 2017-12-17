%function [gradient,dInputs]=neuralModelWithProductComputeDeltaError(x,p,outState,sys)
function [gradient,dInputs]=neuralModelWithProductComputeDeltaError(outState)

global dynamicSystem learning

%sx=size(x);
sx=size(dynamicSystem.state,1);

if isempty(outState)
    %[gradient.outNet,dI]=feval(sys.outNet.backwardFunction,p.outNet,outState.outNetState,outState.delta.*x(1,:));
    [gradient.outNet,dI]=feval(dynamicSystem.config.outNet.backwardFunction,dynamicSystem.parameters.outNet,...
        learning.current.outState.outNetState,learning.current.outState.delta.*dynamicSystem.state(1,:));
    %dInputs(1,:)=dI(1,:)+outState.delta .* outState.outNetState.outs;
    dInputs(1,:)=dI(1,:)+learning.current.outState.delta .* learning.current.outState.outNetState.outs;
else
     %[gradient.outNet,dI]=feval(sys.outNet.backwardFunction,p.outNet,outState.outNetState,outState.delta.*x(1,:));
    [gradient.outNet,dI]=feval(dynamicSystem.config.outNet.backwardFunction,dynamicSystem.parameters.outNet,...
        outState.outNetState,outState.delta.*dynamicSystem.state(1,:));
    %dInputs(1,:)=dI(1,:)+outState.delta .* outState.outNetState.outs;
    dInputs(1,:)=dI(1,:)+outState.delta .* outState.outNetState.outs;
end

%dInputs(2:sx(1),:)=dI(2:sx(1),:);
dInputs(2:sx,:)=dI(2:sx,:);