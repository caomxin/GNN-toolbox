function x=mytest

global dataSet dynamicSystem learning

[dynamicSystem.state,learning.current.forwardState,learning.current.forwardIt]=feval(dynamicSystem.config.forwardFunction,...
        learning.config.maxForwardSteps,dynamicSystem.state,'trainSet',0);
    
x=dynamicSystem.state;
