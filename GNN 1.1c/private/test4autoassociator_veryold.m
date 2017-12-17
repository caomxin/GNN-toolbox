function test4autoassociator
% testing function for classification problems (private)

global dataSet dynamicSystem learning testing TestFigH diffcur diffop
    

dataSet.testSet.forwardSteps=30;

supervisedNodesNumberTrain=size(find(diag(dataSet.trainSet.maskMatrix)),1);
supervisedNodesNumberTest=size(find(diag(dataSet.testSet.maskMatrix)),1);

%cerco la soglia migliore sul validation Set con parametri ottimi
learning.current.validationState=feval(dynamicSystem.config.forwardFunction,learning.config.maxStepsForValidation,learning.current.validationState,...
            'validationSet',1);

[learning.current.validationError learning.current.validationOut]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',[],1);

in=[ learning.current.validationState; dataSet.validationSet.nodeLabels];
Threshold=0;
corrtmp=0;

for Threshold=0:dataSet.config.ThresholdPass:dataSet.config.ThresholdUp
    corretti=0; 
    learning.current.validationMistakenPatternIndex=[];
    for i=1:size(learning.current.validationOut.outNetState.outs,2)
        diff= ( learning.current.validationOut.outNetState.outs(:,i)-in(:,i) )'*( learning.current.validationOut.outNetState.outs(:,i) - in(:,i));
        t=dataSet.validationSet.targets(i);
        if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
            corretti=corretti+1;
        else 
            learning.current.validationMistakenPatternIndex=[learning.current.validationMistakenPatternIndex,i];
            %if (t>0) errpos=errpos+1;
            %else errneg=errneg+1;
            %end
        end
    end
    if corretti>corrtmp
        corrtmp=corretti;
        thretmp=Threshold;
        patttmp=learning.current.validationMistakenPatternIndex;
    end
end
learning.current.validationMistakenPatternIndex=patttmp;
Threshold=thretmp
corretti=corrtmp;  



%valuto il train current con la soglia trovata
[x,currentTrainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',0);
% [testing.current.trainSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.trainSet,...
%     dynamicSystem.parameters,dynamicSystem.config);
[testing.current.trainSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,0);

in=[x;dataSet.trainSet.nodeLabels];  %GABRIELE

corretti=0;

    testing.current.trainSet.mistakenPatternIndex=[];
    for i=1:size(currentTrainOutState.outNetState.outs,2)
        diff= ( currentTrainOutState.outNetState.outs(:,i)-in(:,i) )'*( currentTrainOutState.outNetState.outs(:,i) - in(:,i));
        t=dataSet.trainSet.targets(i);
        diffcur(i)=diff;     
        if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
           corretti=corretti+1;
        else testing.current.trainSet.mistakenPatternIndex=[testing.current.trainSet.mistakenPatternIndex,i];
        end
    end

%% save outputs, useful to evaluate equal error rate
testing.current.trainSet.out=currentTrainOutState.outNetState.outs;

testing.optimal.trainSet.mistakenPatterns=dataSet.trainSet.nodeLabels(:,testing.current.trainSet.mistakenPatternIndex);
testing.current.trainSet.mistakenTargets=dataSet.trainSet.targets(testing.current.trainSet.mistakenPatternIndex);
testing.current.trainSet.accuracy=corretti/size(currentTrainOutState.outNetState.outs,2);

% evaluating optimal parameters on trainSet
% x=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,...
%     dataSet.trainSet, learning.current.optimalParameters,dynamicSystem.config);
[x,trainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,'trainSet',1);

% [testing.optimal.trainSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.trainSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
[testing.optimal.trainSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,'trainSet',x,1);

in=[x;dataSet.trainSet.nodeLabels];

corretti=0;
testing.optimal.trainSet.mistakenPatternIndex=[];
for i=1:size(trainOutState.outNetState.outs,2)
   diff= ( trainOutState.outNetState.outs(:,i)-in(:,i) )'*( trainOutState.outNetState.outs(:,i) - in(:,i));
   t=dataSet.trainSet.targets(i);  %GABRIELE
   if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
       corretti=corretti+1;
   else testing.optimal.trainSet.mistakenPatternIndex=[testing.optimal.trainSet.mistakenPatternIndex,i];
   end
  
end

%% save outputs, useful to evaluate equal error rate
testing.optimal.trainSet.out=trainOutState.outNetState.outs;

%testing.optimal.trainSet.mistakenPatterns=


testing.optimal.trainSet.mistakenPatterns=dataSet.trainSet.nodeLabels(:,testing.optimal.trainSet.mistakenPatternIndex);
testing.optimal.trainSet.mistakenTargets=dataSet.trainSet.targets(testing.optimal.trainSet.mistakenPatternIndex);
testing.optimal.trainSet.accuracy=corretti/size(trainOutState.outNetState.outs,2);


% evaluating current parameters on testSet
% x=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
%     zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),dataSet.testSet,dynamicSystem.parameters,dynamicSystem.config);
[x,currentTestForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
    zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),'testSet',0);
% [testing.current.testSet.error,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.testSet,...
%     dynamicSystem.parameters,dynamicSystem.config);
[testing.current.testSet.error,currentTestOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,0);

in=[x;dataSet.testSet.nodeLabels];

corretti=0;
testing.current.testSet.mistakenPatternIndex=[];
for i=1:size(currentTestOutState.outNetState.outs,2)
   diff= ( currentTestOutState.outNetState.outs(:,i)-in(:,i) )'*( currentTestOutState.outNetState.outs(:,i) - in(:,i));
   t=dataSet.testSet.targets(i);
   if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
       corretti=corretti+1;
   else testing.current.testSet.mistakenPatternIndex=[testing.current.testSet.mistakenPatternIndex,i];
   end
end


%% save outputs, useful to evaluate equal error rate
testing.current.testSet.out=currentTestOutState.outNetState.outs;

testing.current.testSet.mistakenPatterns=dataSet.testSet.nodeLabels(:,testing.current.testSet.mistakenPatternIndex);
testing.current.testSet.mistakenTargets=dataSet.testSet.targets(testing.current.testSet.mistakenPatternIndex);
testing.current.testSet.accuracy=corretti/size(currentTestOutState.outNetState.outs,2);


% evaluating optimal parameters on testSet
% [x,testForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,...
%     zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),dataSet.testSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
[x,testForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.testSet.nNodes),...
    'testSet',1);
% [testing.optimal.testSet.error,testOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.testSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
[testing.optimal.testSet.error,testOutState]=feval(dynamicSystem.config.computeErrorFunction,'testSet',x,1);

in=[x;dataSet.testSet.nodeLabels];

corretti=0;
testing.optimal.testSet.mistakenPatternIndex=[];

for i=1:size(testOutState.outNetState.outs,2)
   diff= (testOutState.outNetState.outs(:,i)-in(:,i) )'*( testOutState.outNetState.outs(:,i) - in(:,i));
   t=dataSet.testSet.targets(i);
   if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
       corretti=corretti+1;
   else testing.optimal.testSet.mistakenPatternIndex=[testing.optimal.testSet.mistakenPatternIndex, i];  
   end
end
%% save outputs, useful to evaluate equal error rate
testing.optimal.testSet.out=testOutState.outNetState.outs;
    
testing.optimal.testSet.mistakenPatterns=dataSet.testSet.nodeLabels(:,testing.optimal.testSet.mistakenPatternIndex);
testing.optimal.testSet.mistakenTargets=dataSet.testSet.targets(testing.optimal.testSet.mistakenPatternIndex);
testing.optimal.testSet.accuracy=corretti/size(testOutState.outNetState.outs,2);

% evaluate accuracy on graphs (instead of on nodes)
if isfield(dataSet.trainSet,'graphNum')&&isfield(dataSet.config,'graphDim')
    if dataSet.trainSet.graphNum > 1
        % it makes sense to evaluate accuracy on graphs
        a={'trainSet','testSet'};
        b={'optimal','current'};
        for ai=1:2
            for bi=1:2
                eval(['err=zeros(size(dataSet.' a{ai} '.targets));']);
                eval(['err(testing.' b{bi} '.' a{ai} '.mistakenPatternIndex)=1;']);
                eval(['g_err=reshape(err,dataSet.config.graphDim,dataSet.' a{ai} '.graphNum);']);
                eval(['testing.' b{bi} '.' a{ai} '.accuracyOnGraphs=size(find(sum(g_err)==0),2)/dataSet.' a{ai} '.graphNum;']);
            end
        end
    end
end

% displays results
global VisualMode
if VisualMode == 1
    if isfield(testing.current.trainSet, 'accuracyOnGraphs')
        TestFigH=DisplayTestC(testing.current.trainSet.error,testing.current.trainSet.accuracy,...
            testing.optimal.trainSet.error,testing.optimal.trainSet.accuracy,...
            testing.current.testSet.error,testing.current.testSet.accuracy,...
            testing.optimal.testSet.error,testing.optimal.testSet.accuracy,...
            testing.current.trainSet.accuracyOnGraphs, testing.optimal.trainSet.accuracyOnGraphs,...
            testing.current.testSet.accuracyOnGraphs, testing.optimal.testSet.accuracyOnGraphs);
    else
        TestFigH=DisplayTestC(testing.current.trainSet.error, testing.current.trainSet.accuracy,...
            testing.optimal.trainSet.error, testing.optimal.trainSet.accuracy,...
            testing.current.testSet.error,testing.current.testSet.accuracy,...
            testing.optimal.testSet.error,testing.optimal.testSet.accuracy);
    end
    
else
    message1(sprintf('\n\t\t\tTESTSET\t\t\t\t\t\tTRAINSET'));
    message1(sprintf('-----------------------------------------------------------------------------------------------'));
    message1([sprintf('\t\t| Accuracy: \t\t') num2str(testing.optimal.testSet.accuracy*100,'%4.2f') '%%' sprintf('\t\t\tAccuracy: \t\t') num2str(testing.optimal.trainSet.accuracy*100,'%4.2f') '%%']);
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
        message1([sprintf('\t\t| Accuracy on graphs: \t') num2str(testing.optimal.testSet.accuracyOnGraphs*100,'%4.2f') '%%' sprintf('\t\t\tAccuracy on graphs: \t') num2str(testing.optimal.trainSet.accuracyOnGraphs*100,'%4.2f') '%%']);
    end
    message1(['OPTIMAL' sprintf('\t\t| Error: \t\t') num2str(testing.optimal.testSet.error,'%10.5g') sprintf('\t\t\tError: \t\t\t') num2str(testing.optimal.trainSet.error,'%10.5g')]);
    message1(sprintf('\t\t|'));
    message1(sprintf('\t\t|'));
    message1([sprintf('\t\t| Accuracy: \t\t') num2str(testing.current.testSet.accuracy*100,'%4.2f') '%%' sprintf('\t\t\tAccuracy: \t\t') num2str(testing.current.trainSet.accuracy*100,'%4.2f') '%%']);
    if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
       message1([sprintf('\t\t| Accuracy on graphs: \t') num2str(testing.current.testSet.accuracyOnGraphs*100,'%4.2f') '%%' sprintf('\t\t\tAccuracy on graphs: \t') num2str(testing.current.trainSet.accuracyOnGraphs*100,'%4.2f') '%%']);
    end
    message1(['CURRENT' sprintf('\t\t| Error: \t\t') num2str(testing.current.testSet.error,'%10.5g') sprintf('\t\t\tError: \t\t\t') num2str(testing.current.trainSet.error,'%10.5g')]);
    message1(sprintf('-----------------------------------------------------------------------------------------------'));
end
