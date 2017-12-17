function fixHandles
global dynamicSystem

dynamicSystem.config.computeErrorFunction=str2func(func2str(dynamicSystem.config.computeErrorFunction));
dynamicSystem.config.computeDeltaErrorFunction=str2func(func2str(dynamicSystem.config.computeDeltaErrorFunction));
dynamicSystem.config.forwardFunction=str2func(func2str(dynamicSystem.config.forwardFunction));
dynamicSystem.config.backwardFunction=str2func(func2str(dynamicSystem.config.backwardFunction));
dynamicSystem.config.forwardJacobianFunction=str2func(func2str(dynamicSystem.config.forwardJacobianFunction));
dynamicSystem.config.backwardJacobianFunction=str2func(func2str(dynamicSystem.config.backwardJacobianFunction));

dynamicSystem.config.transitionNet.forwardFunction=str2func(func2str(dynamicSystem.config.transitionNet.forwardFunction));
dynamicSystem.config.transitionNet.backwardFunction=str2func(func2str(dynamicSystem.config.transitionNet.backwardFunction));
dynamicSystem.config.transitionNet.getDeltaJacobianFunction=str2func(func2str(dynamicSystem.config.transitionNet.getDeltaJacobianFunction));

dynamicSystem.config.outNet.forwardFunction=str2func(func2str(dynamicSystem.config.outNet.forwardFunction));
dynamicSystem.config.outNet.backwardFunction=str2func(func2str(dynamicSystem.config.outNet.backwardFunction));
dynamicSystem.config.outNet.getDeltaJacobianFunction=str2func(func2str(dynamicSystem.config.outNet.getDeltaJacobianFunction));
