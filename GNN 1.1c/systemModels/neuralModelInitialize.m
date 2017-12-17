function neuralModelInitialize
% Initialization function for the neural model

% This function initializes transition neural networks and output neural network
% The parameters involved are automatically computed on the base
% of the system and the trainset parameters: do not manually edit
global dynamicSystem dataSet

dynamicSystem.config.outNet.nInputs=dynamicSystem.config.nStates+dataSet.config.nodeLabelsDim;
dynamicSystem.config.outNet.nOuts=dynamicSystem.config.nOuts;
[dynamicSystem.parameters.outNet,dynamicSystem.config.outNet]=initializeNet(dynamicSystem.config.outNet);

if isfield(dynamicSystem.config,'useLabelledEdges') && dynamicSystem.config.useLabelledEdges==1
    dynamicSystem.config.transitionNet.nInputs=2*dataSet.config.nodeLabelsDim+dynamicSystem.config.nStates+dataSet.config.edgeLabelsDim;
else
    dynamicSystem.config.transitionNet.nInputs=2*dataSet.config.nodeLabelsDim+dynamicSystem.config.nStates;
end
dynamicSystem.config.transitionNet.nOuts=dynamicSystem.config.nStates;
[dynamicSystem.parameters.transitionNet,dynamicSystem.config.transitionNet]=initializeNet(dynamicSystem.config.transitionNet);

%COMPUTING USEFUL DATA
% E=speye(dynamicSystem.config.nStates); % da riusare tutte le volte che serve
prepareDataset('trainSet');
% [dataSet.trainSet.neuralModel.childOfArc,dataSet.trainSet.neuralModel.fatherOfArc]=find(dataSet.trainSet.connMatrix);
% dataSet.trainSet.nArcs=size(dataSet.trainSet.neuralModel.childOfArc,1);
% dataSet.trainSet.neuralModel.fatherToArcMatrix=logical(kron(sparse(1:dataSet.trainSet.nArcs, ...
%     dataSet.trainSet.neuralModel.fatherOfArc,ones(dataSet.trainSet.nArcs,1),dataSet.trainSet.nArcs,dataSet.trainSet.nNodes),E));
% dataSet.trainSet.neuralModel.childToArcMatrix=logical(kron(sparse(1:dataSet.trainSet.nArcs, ...
%     dataSet.trainSet.neuralModel.childOfArc,ones(dataSet.trainSet.nArcs,1),dataSet.trainSet.nArcs,dataSet.trainSet.nNodes),E));
% [r,c]=find(ones(dynamicSystem.config.nStates,dynamicSystem.config.nStates*dataSet.trainSet.nArcs,'uint8'));
% dataSet.trainSet.neuralModel.toBlockMatrixColumn=c;
% dataSet.trainSet.neuralModel.toBlockMatrixRow=r+floor((c-1)/dynamicSystem.config.nStates)*dynamicSystem.config.nStates;
% dataSet.trainSet.neuralModel.arcOfFatherMatrix=logical(sparse(1:dataSet.trainSet.nArcs, dataSet.trainSet.neuralModel.fatherOfArc,...
%     ones(dataSet.trainSet.nArcs,1),dataSet.trainSet.nArcs,dataSet.trainSet.nNodes));
%dataSet.trainSet.neuralModel.arcOfChildMatrix=logical(sparse(1:dataSet.trainSet.nArcs, dataSet.trainSet.neuralModel.childOfArc,...
%    ones(dataSet.trainSet.nArcs,1),dataSet.trainSet.nArcs,dataSet.trainSet.nNodes));

if isfield(dataSet,'validationSet')
    prepareDataset('validationSet');
%     [dataSet.validationSet.neuralModel.childOfArc,dataSet.validationSet.neuralModel.fatherOfArc]=find(dataSet.validationSet.connMatrix);
%     dataSet.validationSet.nArcs=size(dataSet.validationSet.neuralModel.childOfArc,1);
%     dataSet.validationSet.neuralModel.fatherToArcMatrix=logical(kron(sparse(1:dataSet.validationSet.nArcs,...
%         dataSet.validationSet.neuralModel.fatherOfArc,ones(dataSet.validationSet.nArcs,1),dataSet.validationSet.nArcs,...
%         dataSet.validationSet.nNodes),E));
%     dataSet.validationSet.neuralModel.childToArcMatrix=logical(kron(sparse(1:dataSet.validationSet.nArcs,...
%         dataSet.validationSet.neuralModel.childOfArc,ones(dataSet.validationSet.nArcs,1),dataSet.validationSet.nArcs,...
%         dataSet.validationSet.nNodes),E));
%     [r,c]=find(ones(dynamicSystem.config.nStates,dynamicSystem.config.nStates*dataSet.validationSet.nArcs,'uint8'));
%     dataSet.validationSet.neuralModel.toBlockMatrixColumn=c;
%     dataSet.validationSet.neuralModel.toBlockMatrixRow=r+floor((c-1)/dynamicSystem.config.nStates)*dynamicSystem.config.nStates;
%     dataSet.validationSet.neuralModel.arcOfFatherMatrix=logical(sparse(1:dataSet.validationSet.nArcs,...
%         dataSet.validationSet.neuralModel.fatherOfArc,ones(dataSet.validationSet.nArcs,1),dataSet.validationSet.nArcs,...
%         dataSet.validationSet.nNodes));
    %dataSet.validationSet.neuralModel.arcOfChildMatrix=logical(sparse(1:dataSet.validationSet.nArcs,...
    %    dataSet.validationSet.neuralModel.childOfArc,ones(dataSet.validationSet.nArcs,1),dataSet.validationSet.nArcs,...
    %    dataSet.validationSet.nNodes));
end

if isfield(dataSet,'testSet')
    prepareDataset('testSet');
%     [dataSet.testSet.neuralModel.childOfArc,dataSet.testSet.neuralModel.fatherOfArc]=find(dataSet.testSet.connMatrix);
%     dataSet.testSet.nArcs=size(dataSet.testSet.neuralModel.childOfArc,1);
%     dataSet.testSet.neuralModel.fatherToArcMatrix=logical(kron(sparse(1:dataSet.testSet.nArcs,...
%         dataSet.testSet.neuralModel.fatherOfArc,ones(dataSet.testSet.nArcs,1),dataSet.testSet.nArcs,dataSet.testSet.nNodes),E));
%     dataSet.testSet.neuralModel.childToArcMatrix=logical(kron(sparse(1:dataSet.testSet.nArcs,...
%         dataSet.testSet.neuralModel.childOfArc,ones(dataSet.testSet.nArcs,1),dataSet.testSet.nArcs,dataSet.testSet.nNodes),E));
%     [r,c]=find(ones(dynamicSystem.config.nStates,dynamicSystem.config.nStates*dataSet.testSet.nArcs,'uint8'));
%     dataSet.testSet.neuralModel.toBlockMatrixColumn=c;
%     dataSet.testSet.neuralModel.toBlockMatrixRow=r+floor((c-1)/dynamicSystem.config.nStates)*dynamicSystem.config.nStates;
%     dataSet.testSet.neuralModel.arcOfFatherMatrix=logical(sparse(1:dataSet.testSet.nArcs, dataSet.testSet.neuralModel.fatherOfArc,...
%         ones(dataSet.testSet.nArcs,1),dataSet.testSet.nArcs,dataSet.testSet.nNodes));
%     %dataSet.testSet.neuralModel.arcOfChildMatrix=logical(sparse(1:dataSet.testSet.nArcs, dataSet.testSet.neuralModel.childOfArc,...
%     %    ones(dataSet.testSet.nArcs,1),dataSet.testSet.nArcs,dataSet.testSet.nNodes));
end

%%%%%%%%%%%%%%%%%%%%%%%%% OPTIMIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%
%childToArcMatrix                   --> logical
%fatherToArcMatrix                  --> logical

%arcOfFatherMatrix                  --> logical
%arcOfChildMatrix                   --> logical e commentata perch? non ? riusata

%eye(dynamicSyste.config.nStates)   --> speye(dynamicSyste.config.nStates) e allocata una volta per tutte

%find(ones(...))                    --> find(ones(...,'uint8')) cos? la matrice temporanea ? pi? piccola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

