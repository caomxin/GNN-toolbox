
%% This procedure check whether the computation of the the
%% Jacobian is correct. It computes the Jacobian by getJacobian and by the
%% small perturbation method and compares the results.

global dataSet dynamicSystem

%& First system is run in order to be sure that is became stable
stepsForStability=100;
%[x0,fs0]=feval(dynamicSystem.config.forwardFunction,stepsForStability,dynamicSystem.state,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
[x0,fs0]=feval(dynamicSystem.config.forwardFunction,stepsForStability,dynamicSystem.state,'trainSet',0);

%% Then we run again the system for a given number of steps in order to
%% compute the error. Such an error will be compared

%[x,fs]=feval(dynamicSystem.forwardFunction,1,x0,trainSet,dynamicSystem.parameters,dynamicSystem);
[x,fs]=feval(dynamicSystem.config.forwardFunction,1,x0,'trainSet',0);

%[analyticJacobian,analyliticJacobianErrors]=feval(dynamicSystem.config.forwardJacobianFunction, dataSet.trainSet,dynamicSystem.parameters,fs0,dynamicSystem.config);
[analyticJacobian,analyliticJacobianErrors]=feval(dynamicSystem.config.forwardJacobianFunction,'trainSet',fs);



%%
%% computing the javcobian by small perturbations
smallNumber=1e-9;


numExamples=size(x0,2);
sx=size(x0,1);
numUsedExamples=numExamples;
experimentalJacobian=sparse([],[],[],numUsedExamples*sx,numUsedExamples*sx);

for ex=1:numUsedExamples
    for  st=1:sx
        %x=x0;
        %x(st,ex)=x(st,ex)+smallNumber;
        xp=x;
        xp(st,ex)=xp(st,ex)+smallNumber;
        
        %[x2,fs]=feval(dynamicSystem.config.forwardFunction,1,x,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
        [x2,fs]=feval(dynamicSystem.config.forwardFunction,1,xp,'trainSet',0);
        jDiff=x2-x;    %x0??
        usedJD=jDiff(:,1:numUsedExamples);

        experimentalJacobian(:,(ex-1)*sx+st)=usedJD(:)/smallNumber;
    end
end


analyticJacobian=analyticJacobian(1:numUsedExamples*sx,1:numUsedExamples*sx);

analyticNorm=norm(analyticJacobian,1);
experimentalNorm=norm(experimentalJacobian,1);
maxError=norm(analyticJacobian -experimentalJacobian,inf);
relativeErrorWRTAnalytic=norm(analyticJacobian -experimentalJacobian,1)/ analyticNorm;
relativeErrorWRTExperimental=norm(analyticJacobian -experimentalJacobian,1)/ experimentalNorm;

analyticNorm
experimentalNorm
maxError
relativeErrorWRTAnalytic
relativeErrorWRTExperimental




