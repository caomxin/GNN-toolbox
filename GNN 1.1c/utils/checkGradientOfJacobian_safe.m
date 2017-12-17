
%% This procedure check whether the computation of the gradient of the
%% Jacobian is correct. It computes the gradient by getDeltaJacobian and by the
%% small perturbation method and compares the results.

%& First system is run in order to be sure that is became stable
stepsForStability=30;
[x0,fs0]=feval(dynamicSystem.forwardFunction,stepsForStability,dynamicSystem.state,trainSet,dynamicSystem.parameters,dynamicSystem);


%% Then we run again the system for a given number of steps in order to
%% compute the error. Such an error will be compared 

%[x,fs]=feval(dynamicSystem.forwardFunction,1,x0,trainSet,dynamicSystem.parameters,dynamicSystem);

[jacobian,jacobianErrors]=feval(dynamicSystem.forwardJacobianFunction, trainSet,dynamicSystem.parameters,fs0,dynamicSystem);
    
analyticGradient=feval(dynamicSystem.backwardJacobianFunction,trainSet,dynamicSystem.parameters,fs0,jacobian,jacobianErrors,dynamicSystem);
            
error=full(sum(jacobianErrors .^2))/2;

smallNumber=10e-10   ;


clear experimentalGradient;
%%
%% computing the gradient by small perturbations

    for it=fieldnames(dynamicSystem.parameters.transitionNet)'
             sp=size(dynamicSystem.parameters.transitionNet.(char(it)));
           
            for r=1:sp(1)
                for c=1:sp(2)
                    P=dynamicSystem.parameters;
                    P.transitionNet.(char(it))(r,c)=P.transitionNet.(char(it))(r,c)+smallNumber;
                    [x,fs]=feval(dynamicSystem.forwardFunction,1,x0,trainSet,P,dynamicSystem);

                    [xj,xes]=feval(dynamicSystem.forwardJacobianFunction, trainSet,P,fs,dynamicSystem);
                    experimentalError=full(sum(xes .^2))/2;

                  experimentalGradient.(char(it))(r,c)=(experimentalError-error)/smallNumber;                    
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
    it
        gradientDifference.(char(it))=experimentalGradient.(char(it))-analyticGradient.(char(it));
        maxDiff=max(maxDiff,max(max(abs(gradientDifference.(char(it))))));
        sumDiff=sum(sum(abs(gradientDifference.(char(it)))))+sumDiff;
        
        if (sum(experimentalGradient.(char(it))==0)>0),
            fprintf(1,'Some components of experimental gradient are null .. they have not been cosidered');
            consider=experimentalGradient.(char(it))~=0;
        else
            consider=ones(size(experimentalGradient.(char(it))));
        end
        relativeErrorWRTExperimental=max(max(abs(consider .*gradientDifference.(char(it))) ./abs(experimentalGradient.(char(it)))));
        maxRelativeErrorWRTExperimental=max(maxRelativeErrorWRTExperimental,relativeErrorWRTExperimental);
        if (sum(analyticGradient.(char(it))==0)>0),
            fprintf(1,'Some components of analytic gradient are null .. they have not been not cosidered');
            consider=analyticGradient.(char(it))~=0;
        else
            consider=ones(size(analyticGradient.(char(it))));
        end
        relativeErrorWRTAnalytic=max(max(abs(consider .* gradientDifference.(char(it))) ./abs(analyticGradient.(char(it)))));
        maxRelativeErrorWRTAnalytic=max(maxRelativeErrorWRTAnalytic,relativeErrorWRTAnalytic);
  
        
        experimentalGradientNorm=sum(sum(abs(experimentalGradient.(char(it)))))+experimentalGradientNorm;
        analyticGradientNorm=sum(sum(abs(analyticGradient.(char(it)))))+analyticGradientNorm;
end

maxDiff
sumDiff
experimentalGradientNorm
analyticGradientNorm
maxRelativeErrorWRTExperimental
maxRelativeErrorWRTAnalytic
globalRelativeErrorWRTExperimental= sumDiff / experimentalGradientNorm
globalRelativeErrorWRTAnalytic=sumDiff /analyticGradientNorm
