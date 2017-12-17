%% Since it is not easy to write the gradient computation (runBackward) without errrors,
%% this procedure checks gradient computation in order to verify that
%% runBackward does the right calculation w.r.t. the function computed by runForward.
%% The test is carried by introducing a slight change in each parameter of the dynamic system and
%% measuring the difference in the error. 
%% The procedure may be time comsuming, since it has to run the dynamic system as many times as 
%% many parameters the system has.
%% On the other hand, the procedure does not depend on the implementation
%% of runForward and runBakword. Thus, the procedure can be used even after
%% the implementation of the system is changed.


global dynamicSystem dataSet

%& First system is run in order to be sure that is became stable
stepsForStability=100;
%x0=feval(dynamicSystem.config.forwardFunction,stepsForStability,dynamicSystem.state,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
x0=feval(dynamicSystem.config.forwardFunction,stepsForStability,dynamicSystem.state,'trainSet',0);


%% Then we run again the system for a given number of steps in order to
%% compute the error. Such an error will be compared 

stepsForCheck=2;
%[x,fs]=feval(dynamicSystem.config.forwardFunction,stepsForCheck,x0,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
[x,fs]=feval(dynamicSystem.config.forwardFunction,stepsForCheck,x0,'trainSet',0);
%[error,os]=feval(dynamicSystem.config.computeErrorFunction,x0,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
[error,os]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,0);
%[outGradient,deltaX]=feval(dynamicSystem.config.computeDeltaErrorFunction,x,dynamicSystem.parameters,os,dynamicSystem.config);
[outGradient,deltaX]=feval(dynamicSystem.config.computeDeltaErrorFunction,os);
%[gradient,analyticalDX]=feval(dynamicSystem.config.backwardFunction,stepsForCheck,x,dataSet.trainSet,dynamicSystem.parameters,deltaX,fs,dynamicSystem.config);
[gradient,analyticalDX]=feval(dynamicSystem.config.backwardFunction,deltaX,fs,stepsForCheck);

analyticGradient.transitionNet=gradient.transitionNet;
if strcmp(dynamicSystem.config.type,'linear'),
    analyticGradient.forcingNet=gradient.forcingNet;
end
analyticGradient.outNet=outGradient.outNet;

smallNumber=1e-6;

%%
%% computing the gradient by small perturbations

    for it1=fieldnames(dynamicSystem.parameters)'
        for it2=fieldnames(dynamicSystem.parameters.(char(it1)))'
            sp=size(dynamicSystem.parameters.(char(it1)).(char(it2)));
            for r=1:sp(1)
                for c=1:sp(2)

                    P=dynamicSystem.parameters;
                    %P.(char(it1)).(char(it2))(r,c)=P.(char(it1)).(char(it2))(r,c)+smallNumber;
                    dynamicSystem.parameters.(char(it1)).(char(it2))(r,c)=dynamicSystem.parameters.(char(it1)).(char(it2))(r,c)+smallNumber;
                    %xe=feval(dynamicSystem.config.forwardFunction,stepsForCheck,x0,dataSet.trainSet,P,dynamicSystem.config);
                    [xe,niente,ii]=feval(dynamicSystem.config.forwardFunction,stepsForCheck,x0,'trainSet',0);
                     stateDiff.(char(it1)).(char(it2))(r,c)=max(max(abs(xe-x)));
                    %experimentalError=feval(dynamicSystem.config.computeErrorFunction,xe,dataSet.trainSet,P,dynamicSystem.config);
                    experimentalError=feval(dynamicSystem.config.computeErrorFunction,'trainSet',xe,0);
                    experimentalGradient.(char(it1)).(char(it2))(r,c)=(experimentalError-error)/smallNumber;
                    dynamicSystem.parameters=P;
                end
            end        
        end
    end
    
%%
%% computing deltaX by small perturbation 

sx=size(x);
experimentalDX=zeros(sx(1),sx(2));
for r=1:sx(1)
    for c=1:sx(2)
        cX=x;
        cX(r,c)=cX(r,c)+smallNumber;
        %xe=feval(dynamicSystem.config.forwardFunction,stepsForCheck,cX,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
        xe=feval(dynamicSystem.config.forwardFunction,stepsForCheck,cX,'trainSet',0);
        %experimentalError=feval(dynamicSystem.config.computeErrorFunction,xe,dataSet.trainSet,dynamicSystem.parameters,dynamicSystem.config);
        experimentalError=feval(dynamicSystem.config.computeErrorFunction,'trainSet',xe,0);
        experimentalDX(r,c)=(experimentalError-error)/smallNumber;                    
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

for it1=fieldnames(experimentalGradient)'
    for it2=fieldnames(experimentalGradient.(char(it1)))'
        disp(['+++++ ' char(it1) '.' char(it2) ' ++++++']);
        gradientDifference.(char(it1)).(char(it2))=experimentalGradient.(char(it1)).(char(it2))-analyticGradient.(char(it1)).(char(it2));
        maxDiff=max(maxDiff,max(max(abs(gradientDifference.(char(it1)).(char(it2))))));
        sumDiff=sum(sum(abs(gradientDifference.(char(it1)).(char(it2)))))+sumDiff;
        
        relativeErrorWRTExperimental=max(max(abs(gradientDifference.(char(it1)).(char(it2))) ./abs(experimentalGradient.(char(it1)).(char(it2)))));
        maxRelativeErrorWRTExperimental=max(maxRelativeErrorWRTExperimental,relativeErrorWRTExperimental);
        relativeErrorWRTAnalytic=max(max(abs(gradientDifference.(char(it1)).(char(it2))) ./abs(analyticGradient.(char(it1)).(char(it2)))));
        maxRelativeErrorWRTAnalytic=max(maxRelativeErrorWRTAnalytic,relativeErrorWRTAnalytic);
  
        
        componentRelativeErrorWRTExperimental=sum(sum(abs(gradientDifference.(char(it1)).(char(it2))))) /sum(sum(abs(experimentalGradient.(char(it1)).(char(it2)))));
        maxComponentRelativeErrorWRTExperimental=max(maxComponentRelativeErrorWRTExperimental,componentRelativeErrorWRTExperimental);
        componentRelativeErrorWRTAnalytic=max(max(abs(gradientDifference.(char(it1)).(char(it2))))) /sum(sum(abs(analyticGradient.(char(it1)).(char(it2)))));
        maxComponentRelativeErrorWRTAnalytic=max(maxComponentRelativeErrorWRTAnalytic,componentRelativeErrorWRTAnalytic);

        
        experimentalGradientNorm=sum(sum(abs(experimentalGradient.(char(it1)).(char(it2)))))+experimentalGradientNorm;
        analyticGradientNorm=sum(sum(abs(analyticGradient.(char(it1)).(char(it2)))))+analyticGradientNorm;
        disp(sprintf(['\t relativeErrorWRTExperimental \t\t' num2str(relativeErrorWRTExperimental)]))
        disp(sprintf(['\t relativeErrorWRTAnalytic \t\t' num2str(relativeErrorWRTAnalytic)]))
        disp(sprintf(['\t componentRelativeErrorWRTExperimental \t' num2str(componentRelativeErrorWRTExperimental)]))
        disp(sprintf(['\t componentRelativeErrorWRTAnalytic \t' num2str(componentRelativeErrorWRTAnalytic)]))
        disp(sprintf('\n'))
    end
end

experimentalDXNorm=sum(sum(abs(experimentalDX)));
analyticalDXNorm=sum(sum(abs(analyticalDX)));
globalRelativeErrorWRTExperimental= sumDiff / experimentalGradientNorm;
globalRelativeErrorWRTAnalytic=sumDiff /analyticGradientNorm;

disp(sprintf(['experimentalDXNorm \t\t\t\t' num2str(experimentalDXNorm)]))
disp(sprintf(['analyticalDXNorm \t\t\t\t' num2str(analyticalDXNorm)]))
disp(sprintf(['maxDiff \t\t\t\t\t' num2str(maxDiff)]))
disp(sprintf(['sumDiff \t\t\t\t\t' num2str(sumDiff)]))
disp(sprintf(['experimentalGradientNorm \t\t\t' num2str(experimentalGradientNorm)]))
disp(sprintf(['analyticGradientNorm \t\t\t\t' num2str(analyticGradientNorm)]))
disp(sprintf(['maxRelativeErrorWRTExperimental \t\t' num2str(maxRelativeErrorWRTExperimental)]))
disp(sprintf(['maxRelativeErrorWRTAnalytic \t\t\t' num2str(maxRelativeErrorWRTAnalytic)]))
disp(sprintf(['maxComponentRelativeErrorWRTExperimental \t' num2str(maxComponentRelativeErrorWRTExperimental)]))
disp(sprintf(['maxComponentRelativeErrorWRTAnalytic \t\t' num2str(maxComponentRelativeErrorWRTAnalytic)]))
disp(sprintf(['globalRelativeErrorWRTExperimental \t\t' num2str(globalRelativeErrorWRTExperimental)]))
disp(sprintf(['globalRelativeErrorWRTAnalytic \t\t\t' num2str(globalRelativeErrorWRTAnalytic)]))