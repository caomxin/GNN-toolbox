%function [jacobian,jacobianError]=neuralModelGetJacobian(dataset,p,forwardState,sys)
function [jacobian,jacobianError]=neuralModelGetJacobian(dataset,forwardState)
global dataSet dynamicSystem learning

%xdim=dynamicSystem.config.nStates;
%tdj=sparse(xdim,dataSet.(dataset).nArcs*xdim);

tdj=sparse(dynamicSystem.config.nStates,dataSet.(dataset).nArcs*dynamicSystem.config.nStates);

for i=1:dynamicSystem.config.nStates
    eDelta=sparse(dynamicSystem.config.nStates,dataSet.(dataset).nArcs);
    eDelta(i,:)=1;
    
    if isempty(forwardState)
        [g,dj]= feval(dynamicSystem.config.transitionNet.backwardFunction,dynamicSystem.parameters.transitionNet,...
            learning.current.forwardState.transitionNetState,eDelta);
    else
        [g,dj]= feval(dynamicSystem.config.transitionNet.backwardFunction,dynamicSystem.parameters.transitionNet,...
            forwardState.transitionNetState,eDelta);
    end
    
    %%%%%%%%%%%%%%%%%%%%% OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %xdj=dj(1:xdim,:);
    %tdj(i,:)=xdj(:)';

    %% 1st version
    %tdj(i,:)=reshape(dj(1:xdim,:),1,xdim*dataset.nArcs);
    
    tdj(i,:)=reshape(dj(1:dynamicSystem.config.nStates,:),1,dynamicSystem.config.nStates*dataSet.(dataset).nArcs);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

netJacobian=sparse(dataSet.(dataset).neuralModel.toBlockMatrixRow,dataSet.(dataset).neuralModel.toBlockMatrixColumn,tdj(:),...
    dataSet.(dataset).nArcs*dynamicSystem.config.nStates, dataSet.(dataset).nArcs*dynamicSystem.config.nStates);


jacobian=dataSet.(dataset).neuralModel.childToArcMatrix' *netJacobian *dataSet.(dataset).neuralModel.fatherToArcMatrix;

jacobianSum=sum(abs(jacobian));

jacobianError=(jacobianSum > dynamicSystem.config.jacobianThreshold) .* (jacobianSum - dynamicSystem.config.jacobianThreshold);
   