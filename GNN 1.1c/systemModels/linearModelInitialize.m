function linearModelInitialize
% Initialization function for the linear model

% This function initializes transition, forcing and output neural networks
% The parameters involved are automatically computed on the base
% of the system and the trainset parameters: do not manually edit
global dynamicSystem dataSet

dynamicSystem.config.outNet.nInputs=dynamicSystem.config.nStates+dataSet.config.nodeLabelsDim;
dynamicSystem.config.outNet.nOuts=dynamicSystem.config.nOuts;
[dynamicSystem.parameters.outNet,dynamicSystem.config.outNet]=initializeNet(dynamicSystem.config.outNet);

dynamicSystem.config.transitionNet.nInputs=dataSet.config.nodeLabelsDim;
dynamicSystem.config.transitionNet.nOuts=dynamicSystem.config.nStates^2;
[dynamicSystem.parameters.transitionNet,dynamicSystem.config.transitionNet]=initializeNet(dynamicSystem.config.transitionNet);

dynamicSystem.config.forcingNet.nInputs=dataSet.config.nodeLabelsDim;
dynamicSystem.config.forcingNet.nOuts=dynamicSystem.config.nStates;
[dynamicSystem.parameters.forcingNet,dynamicSystem.config.forcingNet]=initializeNet(dynamicSystem.config.forcingNet);
