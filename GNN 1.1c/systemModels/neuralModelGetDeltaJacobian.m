%function grad=neuralModelGetDeltaJacobian(dataset,p,forwardState,jacobian,jacobianError,sys)
function grad=neuralModelGetDeltaJacobian(dataset,jacobian,jacobianError,forwardState)

global dataSet dynamicSystem learning

%xdim=dynamicSystem.config.nStates;

pe=reshape(jacobianError,dynamicSystem.config.nStates,dataSet.(dataset).nNodes);
pa=dataSet.(dataset).neuralModel.arcOfFatherMatrix * pe';
[arc,j]=find(pa~=0);

if isempty(forwardState)
    base=learning.current.forwardState.transitionNetState.inputs(:,arc); %dataset.neuralModel.fatherOfArc(arc));
else
    base=forwardState.transitionNetState.inputs(:,arc); %dataset.neuralModel.fatherOfArc(arc));
end

for i=1:size(arc,1)
    delta(:,i)=pa(arc(i),j(i))*sign(jacobian((...
        dataSet.(dataset).neuralModel.childOfArc(arc(i))-1)*dynamicSystem.config.nStates+(1:dynamicSystem.config.nStates),...
        (dataSet.(dataset).neuralModel.fatherOfArc(arc(i))-1)*dynamicSystem.config.nStates+j(i)));    
end

%[y,netState]=feval(sys.transitionNet.forwardFunction,base,p.transitionNet);
netState=feval(dynamicSystem.config.transitionNet.forwardFunction,base,'transitionNet',0);

grad=feval(dynamicSystem.config.transitionNet.getDeltaJacobianFunction,dynamicSystem.parameters.transitionNet,netState,delta,j);
