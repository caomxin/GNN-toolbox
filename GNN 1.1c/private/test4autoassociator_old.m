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
f1tmp=0;
for Threshold=0.1:dataSet.config.ThresholdPass/2:dataSet.config.ThresholdUp
    
    f1=0;
    corretti=0;
    corrPos=0;
    corrNeg=0;
    learning.current.validationMistakenPatternIndex=[];
    for i=1:size(learning.current.validationOut.outNetState.outs,2)
        diff= ( learning.current.validationOut.outNetState.outs(:,i)-in(:,i) )'*( learning.current.validationOut.outNetState.outs(:,i) - in(:,i));
        t=dataSet.validationSet.targets(i);
 
       if(diff<Threshold)&(t>0)
          corrPos=corrPos+1;
       end
        
       if(diff>Threshold)&(t<0)
          corrNeg=corrNeg+1;
            
      end
      %  corretti=corrPos+(corrNeg*dynamicSystem.config.validationBalancingFact);
        
        if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
            corretti=corretti+1; 
       
        else 
            learning.current.validationMistakenPatternIndex=[learning.current.validationMistakenPatternIndex,i];
            %if (t>0) errpos=errpos+1;
            %else errneg=errneg+1;
            %end
        end
    end
    tarCur=mistaken2all(learning.current.validationMistakenPatternIndex,dataSet.validationSet.targets);
    quality=QualityParameters(dataSet.validationSet.targets,tarCur);
    
    f1= quality.f1;
    %f1=corrPos*corrNeg;
    
    if f1>f1tmp
        f1tmp=f1;
        corrtmp=corretti;
        thretmp=Threshold;
        patttmp=learning.current.validationMistakenPatternIndex;
    end
   
end
learning.current.validationMistakenPatternIndex=patttmp;
Threshold=thretmp
corretti=corrtmp
f1=f1tmp



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
   
        if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
           corretti=corretti+1;
          % testing.current.trainSet.targets(i)=1;
        else
            testing.current.trainSet.mistakenPatternIndex=[testing.current.trainSet.mistakenPatternIndex,i];
           % testing.current.trainSet.targets(i)=-1;
        end
    end
   tar=mistaken2all(testing.current.trainSet.mistakenPatternIndex,dataSet.trainSet.targets);
   testing.current.trainSet.targets=tar; 
    
%% save outputs, useful to evaluate equal error rate
testing.current.trainSet.out=currentTrainOutState.outNetState.outs;



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
   t=dataSet.trainSet.targets(i);
  
   if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
       corretti=corretti+1;
       %testing.optimal.trainSet.targets(i)=1;
   else
       testing.optimal.trainSet.mistakenPatternIndex=[testing.optimal.trainSet.mistakenPatternIndex,i];
       %testing.optimal.trainSet.targets(i)=-1;
   end
  
end
tar=mistaken2all(testing.optimal.trainSet.mistakenPatternIndex,dataSet.trainSet.targets);
   testing.optimal.trainSet.targets=tar; 
 
%% save outputs, useful to evaluate equal error rate
testing.optimal.trainSet.out=trainOutState.outNetState.outs;

%testing.optimal.trainSet.mistakenPatterns=

testing.optimal.trainSet.accuracy=corretti/size(currentTrainOutState.outNetState.outs,2);

%testing.optimal.trainSet.accuracy=corretti/size(trainOutState.outNetState.outs,2);


%valuto il val current con la soglia trovata
[x,currentTrainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,zeros(dynamicSystem.config.nStates,dataSet.validationSet.nNodes),'validationSet',0);
% [testing.current.trainSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.trainSet,...
%     dynamicSystem.parameters,dynamicSystem.config);
[testing.current.validationSet.error,currentTrainOutState]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',x,0);

in=[x;dataSet.validationSet.nodeLabels];  %GABRIELE

corretti=0;
   
    testing.current.validationSet.mistakenPatternIndex=[];
    for i=1:size(currentTrainOutState.outNetState.outs,2)
        diff= ( currentTrainOutState.outNetState.outs(:,i)-in(:,i) )'*( currentTrainOutState.outNetState.outs(:,i) - in(:,i));
        t=dataSet.validationSet.targets(i);
      
        if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
           corretti=corretti+1;
          % testing.current.trainSet.targets(i)=1;
        else
            testing.current.validationSet.mistakenPatternIndex=[testing.current.validationSet.mistakenPatternIndex,i];
           % testing.current.trainSet.targets(i)=-1;
        end
    end
   tar=mistaken2all(testing.current.validationSet.mistakenPatternIndex,dataSet.validationSet.targets);
   testing.current.validationSet.targets=tar; 
    
%% save outputs, useful to evaluate equal error rate
testing.current.validationSet.out=currentTrainOutState.outNetState.outs;
testing.current.validationSet.accuracy=corretti/size(currentTrainOutState.outNetState.outs,2);
% evaluating optimal parameters on trainSet
% x=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps,dynamicSystem.state,...
%     dataSet.trainSet, learning.current.optimalParameters,dynamicSystem.config);
[x,currentTrainForwardState]=feval(dynamicSystem.config.forwardFunction,dataSet.testSet.forwardSteps, zeros(dynamicSystem.config.nStates,dataSet.validationSet.nNodes),'validationSet',1);

% [testing.optimal.trainSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,x,dataSet.trainSet,...
%     learning.current.optimalParameters,dynamicSystem.config);
[testing.optimal.validationSet.error,trainOutState]=feval(dynamicSystem.config.computeErrorFunction,'validationSet',x,1);

in=[x;dataSet.validationSet.nodeLabels];

corretti=0;
testing.optimal.validationSet.mistakenPatternIndex=[];

for i=1:size(trainOutState.outNetState.outs,2)
   diff= ( trainOutState.outNetState.outs(:,i)-in(:,i) )'*( trainOutState.outNetState.outs(:,i) - in(:,i));
   t=dataSet.validationSet.targets(i);
        %GABRIELE
   if(diff<Threshold)&(t>0)|(diff>Threshold)&(t<0)
       corretti=corretti+1;
       %testing.optimal.trainSet.targets(i)=1;
   else
       testing.optimal.validationSet.mistakenPatternIndex=[testing.optimal.validationSet.mistakenPatternIndex,i];
       %testing.optimal.trainSet.targets(i)=-1;
   end
  
end
tar=mistaken2all(testing.optimal.validationSet.mistakenPatternIndex,dataSet.validationSet.targets);
   testing.optimal.validationSet.targets=tar; 
 
%% save outputs, useful to evaluate equal error rate
testing.optimal.validationSet.out=trainOutState.outNetState.outs;


testing.optimal.validationSet.accuracy=corretti/size(trainOutState.outNetState.outs,2);


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
tar=[];
for i=1:size(currentTestOutState.outNetState.outs,2)
   diff= ( currentTestOutState.outNetState.outs(:,i)-in(:,i) )'*( currentTestOutState.outNetState.outs(:,i) - in(:,i));
   if(diff<=Threshold)
       d=1;
   else
       d=-1;
   end
   tar=[tar(:);d];
  
end


testing.current.testSet.targets=tar;
    
%% save outputs, useful to evaluate equal error rate
testing.current.testSet.out=currentTestOutState.outNetState.outs;

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
tar=[];
for i=1:size(testOutState.outNetState.outs,2)
   diff= (testOutState.outNetState.outs(:,i)-in(:,i) )'*( testOutState.outNetState.outs(:,i) - in(:,i));
 if(diff<=Threshold)
       d=1;
   else
       d=-1;
   end
   tar=[tar(:);d];

end
testing.optimal.testSet.targets=tar;

%% save outputs, useful to evaluate equal error rate
testing.optimal.testSet.out=testOutState.outNetState.outs;
    

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
    if dynamicSystem.config.useLogFile
        h = fopen(dynamicSystem.config.logFile,'a');
        if h==-1
            warn(0, ['I can''t open log file <' dynamicSystem.config.logFile '>. Logging was disabled.']);
            dynamicSystem.config.useLogFile=0;
        end
    end
    if dynamicSystem.config.useLogFile
        fprintf(h, '\n\tVALSET\t\t\t\t\t\t\t\t\tTRAINSET\n');
        fprintf(h, '------------------------------------------------------------------------------\n');
        fprintf(h, ['\tOptimal Accuracy: ' num2str(testing.optimal.validationSet.accuracy*100,'%4.2f') '%%\t\t\t\tOptimal Accuracy: ' num2str(testing.optimal.trainSet.accuracy*100,'%4.2f') '%%\n' ]);
        if isfield(dataSet.config,'graphDim')&&dataSet.testSet.graphNum > 1
            fprintf(h, ['\tOptimal Accuracy on graphs: ' num2str(testing.optimal.validationSet.accuracyOnGraphs*100,'%4.2f') '%%\t\tOptimal Accuracy on graphs: ' num2str(testing.optimal.trainSet.accuracyOnGraphs*100,'%4.2f') '%%\n']);
        end
        fprintf(h, ['\tOptimal Error: ' num2str(testing.optimal.validationSet.error,'%10.5g') '\t\t\t\t\tOptimal Error: ' num2str(testing.optimal.trainSet.error,'%10.5g') '\n']);
        fprintf(h, '\n');
        fprintf(h, ['\tCurrent Accuracy: ' num2str(testing.current.validationSet.accuracy*100,'%4.2f') '%%\t\t\t\tCurrent Accuracy: ' num2str(testing.current.trainSet.accuracy*100,'%4.2f') '%%\n']);
        if isfield(dataSet.config,'graphDim')&&dataSet.validationSet.graphNum > 1
            fprintf(h, ['\tCurrent Accuracy on graphs: ' num2str(testing.current.validationSet.accuracyOnGraphs*100,'%4.2f') '%%\t\tCurrent Accuracy on graphs: ' num2str(testing.current.trainSet.accuracyOnGraphs*100,'%4.2f') '%%\n']);
        end
        fprintf(h, ['\tCurrent Error: ' num2str(testing.current.validationSet.error,'%10.5g') '\t\t\t\t\tCurrent Error: ' num2str(testing.current.trainSet.error,'%10.5g') '\n']);
        fprintf(h, '------------------------------------------------------------------------------\n\n');
    else
        disp(sprintf('\n\t\t\tVALIDATIONSET\t\t\t\t\t\tTRAINSET'));
        disp(sprintf('-----------------------------------------------------------------------------------------------'));
        disp([sprintf('\t\t| Accuracy: \t\t') num2str(testing.optimal.validationSet.accuracy*100,'%4.2f') '%' sprintf('\t\t\tAccuracy: \t\t') num2str(testing.optimal.trainSet.accuracy*100,'%4.2f') '%']);
        if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
            disp([sprintf('\t\t| Accuracy on graphs: \t') num2str(testing.optimal.validationSet.accuracyOnGraphs*100,'%4.2f') '%' sprintf('\t\t\tAccuracy on graphs: \t') num2str(testing.optimal.trainSet.accuracyOnGraphs*100,'%4.2f') '%']);
        end
        disp(['OPTIMAL' sprintf('\t\t| Error: \t\t') num2str(testing.optimal.validationSet.error,'%10.5g') sprintf('\t\t\tError: \t\t\t') num2str(testing.optimal.trainSet.error,'%10.5g')]);
        disp(sprintf('\t\t|'));
        disp(sprintf('\t\t|'));
        disp([sprintf('\t\t| Accuracy: \t\t') num2str(testing.current.validationSet.accuracy*100,'%4.2f') '%' sprintf('\t\t\tAccuracy: \t\t') num2str(testing.current.trainSet.accuracy*100,'%4.2f') '%']);
        if isfield(dataSet.config,'graphDim')&&dataSet.trainSet.graphNum > 1
            disp([sprintf('\t\t| Accuracy on graphs: \t') num2str(testing.current.validationSet.accuracyOnGraphs*100,'%4.2f') '%' sprintf('\t\t\tAccuracy on graphs: \t') num2str(testing.current.trainSet.accuracyOnGraphs*100,'%4.2f') '%']);
        end
        disp(['CURRENT' sprintf('\t\t| Error: \t\t') num2str(testing.current.validationSet.error,'%10.5g') sprintf('\t\t\tError: \t\t\t') num2str(testing.current.trainSet.error,'%10.5g')]);
        disp(sprintf('-----------------------------------------------------------------------------------------------'));
    end
end
