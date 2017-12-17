%% This procedure check whether the computation of the gradient of the
%% Jacobian is correct. It computes the gradient by getDeltaJacobian and by the
%% small perturbation method and compares the results.

global dynamicSystem dataSet

%& First system is run in order to be sure that is became stable
stepsForStability=100;
%[x0,fs0]=feval(dynamicSystem.config.forwardFunction,stepsForStability,dynamicSystem.state,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
[x0,fs0]=feval(dynamicSystem.config.forwardFunction,stepsForStability,dynamicSystem.state,'trainSet',0);

%% Then we run again the system for a given number of steps in order to
%% compute the error. Such an error will be compared

%[x,fs]=feval(dynamicSystem.forwardFunction,1,x0,trainSet,dynamicSystem.parameters,dynamicSystem);
[x,fs]=feval(dynamicSystem.config.forwardFunction,1,x0,'trainSet',0);

%[jacobian,jacobianErrors]=feval(dynamicSystem.config.forwardJacobianFunction, dataSet.trainSet,dynamicSystem.parameters,fs0,dynamicSystem.config);
[jacobian,jacobianErrors]=feval(dynamicSystem.config.forwardJacobianFunction, 'trainSet',fs);

%analyticGradient=feval(dynamicSystem.config.backwardJacobianFunction,dataSet.trainSet,dynamicSystem.parameters,fs0,jacobian,jacobianErrors,dynamicSystem.config);
analyticGradient=feval(dynamicSystem.config.backwardJacobianFunction,'trainSet',jacobian,jacobianErrors,fs0);

error=full(sum(jacobianErrors.^2))/2;


smallNumber=1e-9;


clear experimentalGradient;
%%
%% computing the gradient by small perturbations

for it=fieldnames(dynamicSystem.parameters.transitionNet)'
    sp=size(dynamicSystem.parameters.transitionNet.(char(it)));

    %dynamicSystem.parameters.transitionNet.(char(it))
    
    for r=1:sp(1)
        for c=1:sp(2)
            P=dynamicSystem.parameters;
            dynamicSystem.parameters.transitionNet.(char(it))(r,c)=dynamicSystem.parameters.transitionNet.(char(it))(r,c)+smallNumber;
            %[x,fs]=feval(dynamicSystem.config.forwardFunction,1,x0,dataSet.trainSet,P,dynamicSystem.config);
            [x,fs]=feval(dynamicSystem.config.forwardFunction,1,x0,'trainSet',0);
            
            %[xj,xes]=feval(dynamicSystem.config.forwardJacobianFunction, dataSet.trainSet,P,fs,dynamicSystem.config);
            [xj,xes]=feval(dynamicSystem.config.forwardJacobianFunction,'trainSet',fs);
            
            experimentalError=full(sum(xes.^2))/2;
            
            
            experimentalGradient.(char(it))(r,c)=(experimentalError-error)/smallNumber;
            
            dynamicSystem.parameters=P;
        end
    end
end



%% comparing gradients


maxDiff=0;
sumDiff=0;
maxRelativeErrorWRTExperimental=0;
maxRelativeErrorWRTAnalytic=0;
maxComponentRelativeErrorWRTExperimental=0;
maxComponentRelativeErrorWRTAnalytic=0;

experimentalGradientNorm=0;
analyticGradientNorm=0;

for it=fieldnames(experimentalGradient)'
    disp(['+++++ ' char(it) ' ++++++']);
    gradientDifference.(char(it))=experimentalGradient.(char(it))-analyticGradient.(char(it));
    maxDiff=max(maxDiff,max(max(abs(gradientDifference.(char(it))))));
    sumDiff=sum(sum(abs(gradientDifference.(char(it)))))+sumDiff;

    if (sum(experimentalGradient.(char(it))==0)>0),
        fprintf(1,'Some components of experimental gradient are null .. they have not been considered\n');
        consider=experimentalGradient.(char(it))~=0;
    else
        consider=ones(size(experimentalGradient.(char(it))));
    end
    relativeErrorWRTExperimental=max(max(abs(consider .*gradientDifference.(char(it))) ./abs(experimentalGradient.(char(it)))));
    maxRelativeErrorWRTExperimental=max(maxRelativeErrorWRTExperimental,relativeErrorWRTExperimental);
    if (sum(analyticGradient.(char(it))==0)>0),
        fprintf(1,'Some components of analytic gradient are null .. they have not been not considered\n');
        consider=analyticGradient.(char(it))~=0;
    else
        consider=ones(size(analyticGradient.(char(it))));
    end
    relativeErrorWRTAnalytic=max(max(abs(consider .* gradientDifference.(char(it))) ./abs(analyticGradient.(char(it)))));
    maxRelativeErrorWRTAnalytic=max(maxRelativeErrorWRTAnalytic,relativeErrorWRTAnalytic);

    experimentalGradientNorm=sum(sum(abs(experimentalGradient.(char(it)))))+experimentalGradientNorm;
    analyticGradientNorm=sum(sum(abs(analyticGradient.(char(it)))))+analyticGradientNorm;
    
    disp(sprintf(['\t relativeErrorWRTExperimental \t\t' num2str(relativeErrorWRTExperimental)]))
    disp(sprintf(['\t relativeErrorWRTAnalytic \t\t' num2str(relativeErrorWRTAnalytic)]))
    disp(sprintf(['\t experimentalGradientNorm \t\t' num2str(experimentalGradientNorm)]))
    disp(sprintf(['\t analyticGradientNorm \t\t\t' num2str(analyticGradientNorm)]))
    disp(sprintf('\n'))
end

globalRelativeErrorWRTExperimental= sumDiff / experimentalGradientNorm;
globalRelativeErrorWRTAnalytic=sumDiff /analyticGradientNorm;
disp(sprintf(['maxDiff \t\t\t\t\t' num2str(maxDiff)]))
disp(sprintf(['sumDiff \t\t\t\t\t' num2str(sumDiff)]))
disp(sprintf(['experimentalGradientNorm \t\t\t' num2str(experimentalGradientNorm)]))
disp(sprintf(['analyticGradientNorm \t\t\t\t' num2str(analyticGradientNorm)]))
disp(sprintf(['maxRelativeErrorWRTExperimental \t\t' num2str(maxRelativeErrorWRTExperimental)]))
disp(sprintf(['maxRelativeErrorWRTAnalytic \t\t\t' num2str(maxRelativeErrorWRTAnalytic)]))
