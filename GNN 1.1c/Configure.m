% NOTE: when there is an error the function <err> displays appropriate
% message and then issues function "error". The error is catch in the
% calling function returning control to the console. This is done in order
% to avoid that "error" function displays a stack of errors tracking back
% its way onto private functions calling.


function Configure(file)
% USAGE: Configure(<file.config>)
% Configure options for GNN

addpath(genpath(pwd));
 try
    global model model_name learn learn_name general general_name dynamicSystem learning dataSet
    if nargin ~= 1
        file='GNN.config';      %default config file
    end
    [fid,message] = fopen(file,'rt');
    if fid == -1
        err(0,['Can''t open file <' file '>']);
    end
    
    checkDataset;
    
    % model: status     <0 not open   1 open  2 closed>
    %        type       <0 unknown    1 linear  2 neural 3 weblin>
    %        nStates
    %        trans: nlayers, outActivationType, nHiddens, weightRange
    %        forc: nlayers, outActivationType, nHiddens, weightRange
    %        out: nlayers, outActivationType, nHiddens, weightRange
    %        jacobianThreshold, jacobianFactorCoeff
    %        saturationThreshold, saturationCoeff
    %        useJacobianControl,useSaturationControl,useLabelledEdges,errorFunction
    dynamicSystem=[];
    learning=[];
    model=[0 0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 0 0  0 0];
    model_name={'status','type','nStates',...
        'transitionNet.nlayers', 'transitionNet.outActivationType', 'transitionNet.nHiddens', 'transitionNet.weightRange',...
        'forcingNet.nlayers', 'forcingNet.outActivationType', 'forcingNet.nHiddens', 'forcingNet.weightRange',...
        'outNet.nlayers', 'outNet.outActivationType', 'outNet.nHiddens', 'outNet.weightRange',...
        'jacobianThreshold','jacobianFactorCoeff','saturationThreshold','saturationCoeff','useJacobianControl','useSaturationControl',...
        'useLabelledEdges','errorFunction'};
    % learn: status     <idem>
    %        learningSteps, maxForwardSteps, maxBackwardSteps, forwardStopCoefficient, backwardStopCoefficient
    %        stepsForValidation, maxStepsForValidation, stopCoefficientForValidation
    %        deltaMax, deltaMin, nablaP, nablaM
    learn=[0 0 0 0 0 0  0 0 0 0 0 0 0 0  0 0 0 0 0  0];
    learn_name={'status', 'learningSteps', 'maxForwardSteps', 'maxBackwardSteps', 'forwardStopCoefficient', 'backwardStopCoefficient',...
        'stepsForValidation', 'maxStepsForValidation', 'deltaMax', 'deltaMin', 'nablaP', 'nablaM','useValidation','useValidationMistakenPatterns',...
        'saveErrorHistory','saveJacobianHistory','saveSaturationHistory','saveStabilityCoefficientHistory','saveIterationHistory',...
        'saveStateHistory'};
    % general: status      <idem>
    general=[0 0 0 0];
    general_name={'status','useLogFile','useAutoSave','useBalancedDataset'};

    delimiter = [';' ' ' 9 13];         %ASCII(9) = TABS, ASCII(13) = carriage return
    numline=0;
    while feof(fid) == 0
        numline=numline+1;
        line = fgetl(fid);
        if isempty(line)
            warn(numline, 'line is empty. Please remove it or comment it');
        elseif line(1)=='#'
            %this is a comment line
            continue;
        elseif (line(1)=='*')
            %this is a section line
            if (size(strfind(line,'MODEL PARAMETERS START'),2))
                check_start_end('m','s',numline);
                model(1)=1;
            elseif (line(1)=='*') && (size(strfind(line,'MODEL PARAMETERS END'),2))
                check_start_end('m','e',numline);
                model(1)=2;
            elseif (line(1)=='*') && (size(strfind(line,'LEARNING PARAMETERS START'),2))
                check_start_end('l','s',numline);
                learn(1)=1;
            elseif (line(1)=='*') && (size(strfind(line,'LEARNING PARAMETERS END'),2))
                check_start_end('l','e',numline);
                learn(1)=2;
            elseif (line(1)=='*') && (size(strfind(line,'GENERAL PARAMETERS START'),2))
                check_start_end('g','s',numline);
                general(1)=1;
            elseif (line(1)=='*') && (size(strfind(line,'GENERAL PARAMETERS END'),2))
                check_start_end('g','e',numline);
                general(1)=2;
            else
                err(numline,'Malformed section line');
            end
        else
            % this is a parameter line
            if model(1)==1
                section = 'm';
            elseif learn(1)==1
                section = 'l';
            elseif general(1)==1
                section = 'g';
            else
                err(numline,'No section appers to be open');
            end
            line = strtok(line, delimiter);
            k = strfind(line,'=');
            if size(k,2)==0
                err(numline,'No ''='' detected');
            elseif (size(k,2)>1)
                err(numline,'More than one "=" detected');
            end
            name = line(1:k-1);
            value = line(k+1:end);
            check_valueok(section,name,value,num2str(numline));
        end
    end
    fclose(fid);
    check_closed;
    ok=check_thereisall;
    if ~ok
        return;
    end
    
    % Assign initial values to some dynamicSystem and learning parameters.
    % Initialization of the model: proper functions are chosen for the nets of the model
    [part,rem]=strtok(dynamicSystem.config.type,'+');

    if strcmp(part,'neural')
        dynamicSystem.config.forwardFunction=@neuralModelRunForward;
        dynamicSystem.config.backwardFunction=@neuralModelRunBackward;
        dynamicSystem.config.forwardJacobianFunction=@neuralModelGetJacobian;
        dynamicSystem.config.backwardJacobianFunction=@neuralModelGetDeltaJacobian;
    elseif strcmp(part,'linear')
        dynamicSystem.config.forwardFunction=@linearModelRunForward;
        dynamicSystem.config.backwardFunction=@linearModelRunBackward;
        dynamicSystem.config.useJacobianControl=0;
    end
    if ~isempty(rem)
        while true
            rem=rem(2:end);
            [part,rem]=strtok(rem,'+');
            if strcmp(part,'outmult')
                warn(0,sprintf(['Using type ' dynamicSystem.config.type ' is deprecated. Set errorFunction=outmult in config file, insted.\nFuture versions may no longer support old way of specifying error function.']));
                dynamicSystem.config.computeErrorFunction=@neuralModelWithProductComputeError;
                dynamicSystem.config.computeDeltaErrorFunction=@neuralModelWithProductComputeDeltaError;
            end
            if strcmp(part,'ranking')
                warn(0,sprintf(['Using type ' dynamicSystem.config.type ' is deprecated. Set errorFunction=ranking in config file, insted.\nFuture versions may no longer support old way of specifying error function.']));
                dynamicSystem.config.computeErrorFunction=@rankingComputeError;
                dynamicSystem.config.computeDeltaErrorFunction=@rankingComputeDeltaError;
            end
            if isempty(rem)
                break;
            end
        end
    end
    %Special case: Half Hot problem (old way)
    if isfield(dataSet.config,'useQuadraticFunctions')
        warn(0,sprintf('Using dataSet.config.useQuadraticFunctions is deprecated. Set errorFunction=quadratic in config file, insted.\nFuture versions may no longer support dataSet.config.useQuadraticFunctions'));
        dynamicSystem.config.computeErrorFunction=@neuralModelQuadraticComputeError;
        dynamicSystem.config.computeDeltaErrorFunction=@neuralModelQuadraticComputeDeltaError;
    end

    if isfield(dataSet.config,'useLabelledEdges')
        warn(0,sprintf('Using dataSet.config.useLabelledEdges is deprecated. Set useLabelledEdges=1 in config file, insted.\nFuture versions may no longer support dataSet.config.useLabelledEdges'));
        dynamicSystem.config.useLabelledEdges=1;
    end
    if strcmp(func2str(dynamicSystem.config.computeErrorFunction),'autoassociatorComputeError')
        dynamicSystem.config.nOuts=dynamicSystem.config.nStates+dataSet.config.nodeLabelsDim;
    else
        dynamicSystem.config.nOuts=size(dataSet.trainSet.targets,1);
    end
    dynamicSystem.state=zeros(dynamicSystem.config.nStates,dataSet.trainSet.nNodes);
    learning.current.bestErrorOnValidation=realmax;
    learning.current.nSteps=1;
    if dynamicSystem.config.useValidation
        learning.history.validationErrorHistory=[];
        learning.current.validationState=zeros(dynamicSystem.config.nStates,dataSet.validationSet.nNodes);
    end
    learning.history.oldX=dynamicSystem.state;

    % Initialize the neural networks of the models
    if ~isempty(strfind(dynamicSystem.config.type,'neural'))
        neuralModelInitialize
    elseif ~isempty(strfind(dynamicSystem.config.type,'linear'))
        linearModelInitialize
    end
    
    for it1=fieldnames(dynamicSystem.parameters)'
        for it2=fieldnames(dynamicSystem.parameters.(char(it1)))'
            learning.current.rProp.delta.(char(it1)).(char(it2)) =0.001*ones(size(dynamicSystem.parameters.(char(it1)).(char(it2))));
            learning.current.rProp.deltaW.(char(it1)).(char(it2)) =zeros(size(dynamicSystem.parameters.(char(it1)).(char(it2))));
            learning.current.rProp.oldGradient.(char(it1)).(char(it2))=zeros(size(dynamicSystem.parameters.(char(it1)).(char(it2))));
        end
    end
    learning.history.oldP=dynamicSystem.parameters;
    optimize;

    fname=datestr(now);
    fname=regexprep(fname, ' ','_');    % replace the blank space with an underscore
    dynamicSystem.config.name=fname;
    if dynamicSystem.config.useAutoSave
        [status,message]=mkdir('autoSave');
    end

    % if useLogFile==1 create logFile in directory <GNN root>/logs and write the first things
    if dynamicSystem.config.useLogFile
        [status,message]=mkdir('logs');      % if the directory already exists, do nothing 
        fname = ['logs/' fname '.log'];
        h=fopen(fname,'w+');
        dynamicSystem.config.logFile = fname;
        fprintf(h, ['**** GNN log file: \t' datestr(now) ' ****\n']);
        [status,host]=system('uname -n');
        if ~status
            fprintf(h, ['PC Name:\t\t\t' host '\n']);
        end
        [status,opsys]=system('uname -s');
        [status,kernelver]=system('uname -r');
        if ~status
            fprintf(h, ['OS:\t\t\t\t\t' opsys(1:end-1) ' (' kernelver(1:end-1) ')\n']);
        end
        version=ver('matlab');
        fprintf(h, ['Matlab version:\t\t' version.Version ' ' version.Release '\n']);

        [fid,message] = fopen(file,'rt');
        if fid == -1
            warn(0,['Can''t reopen file <' file '> to write the log file']);
        else
            fprintf(h, ['\n**** Config file used ****\n']);
            while feof(fid) == 0
                line = fgetl(fid);
                if line(1) ~= '#'
                    fprintf(h, [line '\n']);
                end
            end
            fprintf(h,'\n\n');
        end
        fclose(h);
    end
    aaaaaa=1
    dynamicSystem.config.configured=1;
    aaaaaa=2

%    catch
%      err(0,['Ooops. There was an error! Configuration was not completed.']);
%      cleanup
%      clear global dynamicSystem learning model model_name learn learn_name general general_name
%      return;
 end




% Check if the dataSet appears to be built properly
function checkDataset
global dataSet
if (isempty(dataSet) || ~isfield(dataSet, 'config') || isempty(dataSet.config))
    err(0, 'I can''t find the dataSet or bad dataSet.config');
end
sets={'trainSet'};
items={'connMatrix','maskMatrix','nodeLabels','targets'};

for i=1:size(sets,2)
    if (~isfield(dataSet,sets{i}) || isempty(dataSet.(sets{i})))
        err(0, ['I can''t find the ' sets{i}]);
    end
    for j=1:size(items,2)
        if (~isfield(dataSet.(sets{i}),items{j}) || isempty(dataSet.(sets{i}).(items{j})))
            err(0, ['I can''t find the ' sets{i} '.' items{j}]);
        end
    end
end
if (~isfield(dataSet.config,'nodeLabelsDim'))
    dataSet.config.nodeLabelsDim=size(dataSet.trainSet.nodeLabels,1);
elseif (dataSet.config.nodeLabelsDim~=size(dataSet.trainSet.nodeLabels,1))
        err(0, 'The value in dataSet.config.nodeLabelsDim is different from the number of columns in dataSet.trainSet.nodeLabels');
end





% Check the correctness of starting and ending of the conf sections
function check_start_end(who,what,num)
global model learn general
switch who
    case 'm'
        a='Model Parameters';
        va=model;
        b='Learning Parameters';
        vb=learn;
        c='General Parameters';
        vc=general;
    case 'l'
        a='Learning Parameters';
        va=learn;
        b='Model Parameters';
        vb=model;
        c='General Parameters';
        vc=general;
    case 'g'
        a='General Parameters';
        va=general;
        b='Model Parameters';
        vb=model;
        c='Learning Parameters';
        vc=learn;
end
try
    if what=='s'
        if vb(1)==1
            err(num,['Cannot open a ' a ' Parameters section inside a ' b ' one']);return;
        elseif vc(1)==1
            err(num,['Cannot open a ' a ' Parameters section inside a ' c ' one']);return;
        elseif va(1)==1
            err(num,[a ' section already open']);return;
        elseif va(1)==2
            err(num,[a ' section already closed']);return;
        end
    else
        if vb(1)==1
            err(num,['Cannot close a ' a ' section inside a ' b ' one']);return;
        elseif vc(1)==1
            err(num,['Cannot close a ' a ' section inside a ' c ' one']);return;
        elseif va(1)==0
            err(num,[a ' section has not been opened']);return;
        elseif va(1)==2
            err(num,[a ' section already closed']);return;
        end
    end
catch
    return;
end


% Check the correctness of the couple <parameter,value>
function check_valueok(section,name,value,n)
try
    if isempty(value)
        err(n,['No value specified for <' name '>']);
    end
    global dynamicSystem learning model learn general dataSet
    if section == 'm'
        if (~strcmp(name,'type')) && (model(1)==0)
            err(n,'<type> is the first parameter that should be set');
        end
        switch name
            case 'type'
                if ~isempty(strfind(value,'linear'))
                    model(2) = 1;
                    dynamicSystem.config.type = value;
                elseif ~isempty(strfind(value,'neural'))
                    model(2) = 2;
                    dynamicSystem.config.type = value;
                else
                    err(n,'Unsupported model type');
                end
            case 'nStates'
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<=0)
                    err(n,['Parameter <' name '> should be a positive integer. Check "' value '"']);return;
                end
                dynamicSystem.config.nStates=v;
                model(3)=1;
            case 'transitionNet.nLayers'
                v=str2double(value);
                if v~=1 && v~=2
                    err(n,['Parameter <' name '> should be 1 or 2. Check "' value '"']);return;
                end
                dynamicSystem.config.transitionNet.nLayers=v;
                model(4)=1;
            case 'transitionNet.outActivationType'
                if strcmp(value,'linear')
                    dynamicSystem.config.transitionNet.outActivationType = 'linear';
                elseif strcmp(value,'tanh')
                    dynamicSystem.config.transitionNet.outActivationType = 'tanh';
                else
                    err(n,['Unsupported ' name '. Check "' value '"']);return;
                end
                model(5)=1;
            case 'transitionNet.nHiddens'
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<1)
                    err(n,['Parameter <' name '> should be an integer greater than 1. Check "' value '"']);return;
                end
                dynamicSystem.config.transitionNet.nHiddens=v;
                model(6)=1;
            case 'transitionNet.weightRange'
                v=str2double(value);
                if isnan(v) || v<10^-10 || v >=10
                    err(n,['Parameter <' name '> should be a real number in the interval [e-10,1). Check "' value '"']);return;
                end
                dynamicSystem.config.transitionNet.weightRange=v;
                model(7)=1;
            case 'forcingNet.nLayers'
                if model(2) == 2
                    warn(n, ['Parameter <' name '> is useless in neural model']);
                end
                v=str2double(value);
                if v~=1 && v~=2
                    err(n,['Parameter <' name '> should be 1 or 2. Check "' value '"']);return;
                end
                dynamicSystem.config.forcingNet.nLayers=v;
                model(8)=1;
            case 'forcingNet.outActivationType'
                if model(2) == 2
                    warn(n, ['Parameter <' name '> is useless in neural model']);
                end
                if strcmp(value,'linear')
                    dynamicSystem.config.forcingNet.outActivationType = 'linear';
                elseif strcmp(value,'tanh')
                    dynamicSystem.config.forcingNet.outActivationType = 'tanh';
                else
                    err(n,['Unsupported ' name '. Check "' value '"']);return;
                end
                model(9)=1;
            case 'forcingNet.nHiddens'
                if model(2) == 2
                    warn(n, ['Parameter <' name '> is useless in neural model']);
                end
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<1)
                    err(n,['Parameter <' name '> should be an integer greater than 1. Check "' value '"']);return;
                end
                dynamicSystem.config.forcingNet.nHiddens=v;
                model(10)=1;
            case 'forcingNet.weightRange'
                if model(2) == 2
                    warn(n, ['Parameter <' name '> is useless in neural model']);
                end
                v=str2double(value);
                if isnan(v) || v<10^-10 || v >=1
                    err(n,['Parameter <' name '> should be a real number in the interval [e-10,1). Check "' value '"']);return;
                end
                dynamicSystem.config.forcingNet.weightRange=v;
                model(11)=1;
            case 'outNet.nLayers'
                v=str2double(value);
                if v~=1 && v~=2
                    err(n,['Parameter <' name '> should be 1 or 2. Check "' value '"']);return;
                end
                dynamicSystem.config.outNet.nLayers=v;
                model(12)=1;
            case 'outNet.outActivationType'
                if strcmp(value,'linear')
                    dynamicSystem.config.outNet.outActivationType = 'linear';
                elseif strcmp(value,'tanh')
                    dynamicSystem.config.outNet.outActivationType = 'tanh';
                else
                    err(n,['Unsupported ' name '. Check "' value '"']);return;
                end
                model(13)=1;
            case 'outNet.nHiddens'
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || (v<1)
                    err(n,['Parameter <' name '> should be an integer greater than 1. Check "' value '"']);return;
                end
                dynamicSystem.config.outNet.nHiddens=v;
                model(14)=1;
            case 'outNet.weightRange'
                v=str2double(value);
                if isnan(v) || v<10^-10 || v >=1
                    err(n,['Parameter <' name '> should be a real number in the interval [e-10,1). Check "' value '"']);return;
                end
                dynamicSystem.config.outNet.weightRange=v;
                model(15)=1;
            case 'jacobianThreshold'
                if model(2) == 1
                    warn(n, ['Parameter <' name '> is useless in linear model']);
                end
                v=str2double(value);
                if v<0 || v >=1
                    err(n,['Parameter <' name '> should be a real number in the interval [0,1). Check "' value '"']);return;
                end
                dynamicSystem.config.jacobianThreshold=v;
                model(16)=1;
            case 'jacobianCoeff'
                if model(2) == 1
                    warn(n, ['Parameter <' name '> is useless in linear model']);
                end
                v=str2double(value);
                if  isnan(v) || v<0 || v >1000000
                    err(n,['Parameter <' name '> should be a real number in the interval [0,1000000]. Check "' value '"']);return;
                end
                if (v==0 && model(20)==1 && dynamicSystem.config.useJacobianControl==1)
                   warn(n, ['Parameter ''useJacobianControl''=1 but parameter ''jacobianFactorCoeff''=0. I set ''useJacobianControl'' to 0 for you']);
                   dynamicSystem.config.useJacobianControl=0;
                end
                dynamicSystem.config.jacobianFactorCoeff=v;
                model(17)=1;
            case 'saturationThreshold'
                v=str2double(value);
                if v<0 || v >=1-1e-9
                    err(n,['Parameter <' name '> should be a real number in the interval [0,1-1e-9). Check "' value '"']);return;
                end
                dynamicSystem.config.saturationThreshold=v;
                model(18)=1;
            case 'saturationCoeff'
                v=str2double(value);
                if  isnan(v) || v<0 || v >100
                    err(n,['Parameter <' name '> should be a real number in the interval [0,100]. Check "' value '"']);return;
                end
                if (v==0 && model(21)==1 && dynamicSystem.config.useSaturationControl==1)
                   warn(n, ['Parameter ''useSaturationControl''=1 but parameter ''saturationCoeff''=0. I set <useSaturationControl> to 0 for you']);
                   dynamicSystem.config.useSaturationControl=0;
                end
                dynamicSystem.config.saturationCoeff=v;
                model(19)=1;
            case 'useJacobianControl'
                if model(2) == 1
                    warn(n, ['Parameter <' name '> is useless in linear model']);
                end
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useJacobianControl=0;
                elseif v==1 && model(17)==1 && dynamicSystem.config.jacobianFactorCoeff==0
                    warn(n, ['Parameter ''useJacobianControl''=1 but parameter ''jacobianFactorCoeff''=0. I set <useJacobianControl> to 0 for you']);
                    dynamicSystem.config.useJacobianControl=0;
                else
                    if v==1 dynamicSystem.config.useJacobianControl=1; else dynamicSystem.config.useJacobianControl=0; end
                end
                model(20)=1;
            case 'useSaturationControl'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useSaturationControl=0;
                else
                    if v==1 dynamicSystem.config.useSaturationControl=1; else dynamicSystem.config.useSaturationControl=0; end
                end
                model(21)=1;
            case 'useLabelledEdges'
                v=str2double(value);
                if isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useLabelledEdges=0;
                end
                dynamicSystem.config.useLabelledEdges=v;
                model(22)=1;
            case 'errorFunction'
                if strcmp(value,'mse')
                    % mean square error
                    dynamicSystem.config.computeErrorFunction=@mseComputeError;
                    dynamicSystem.config.computeDeltaErrorFunction=@mseComputeDeltaError;
                elseif strcmp(value,'quadratic')
                    % quadratic function for HalfHot dataset
                    dynamicSystem.config.computeErrorFunction=@neuralModelQuadraticComputeError;
                    dynamicSystem.config.computeDeltaErrorFunction=@neuralModelQuadraticComputeDeltaError;
                elseif strcmp(value,'outmult')
                    % multiplication in the output function for WebPageScoring dataset
                    dynamicSystem.config.computeErrorFunction=@neuralModelWithProductComputeError;
                    dynamicSystem.config.computeDeltaErrorFunction=@neuralModelWithProductComputeDeltaError;
                elseif strcmp(value,'ranking')
                    % error function for ranking problems (thx to Augusto Pucci)
                    dynamicSystem.config.computeErrorFunction=@rankingComputeError;
                    dynamicSystem.config.computeDeltaErrorFunction=@rankingComputeDeltaError;
                elseif strcmp(value,'autoassociator')
                    % error function for ranking problems (thx to Claudio Prudenzi)
                    dynamicSystem.config.computeErrorFunction=@autoassociatorComputeError;
                    dynamicSystem.config.computeDeltaErrorFunction=@autoassociatorComputeDeltaError;    
                else
                    err(n, sprintf(['Error function <' value '> is not supported. Actually supported error functions are:\n\tmse\n\tquadratic\n\toutmult\n\tranking\n\tautoassociator']));return;
                end
                model(23)=1;
            otherwise
                err(n,['Unknown parameter <' name '> in Model Parameters section']);return;
        end
    elseif section == 'l'
        switch name
            case 'learningSteps'
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || v<1
                    err(n,['Parameter <' name '> should be an integer number greater than 0. Check "' value '"']);return;
                end
                learning.config.learningSteps=v;
                learn(2)=1;
            case 'maxForwardSteps'
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || v<1
                    err(n,['Parameter <' name '> should be an integer number greater than 0. Check "' value '"']);return;
                end
                learning.config.maxForwardSteps=v;
                learn(3)=1;
            case 'maxBackwardSteps'
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || v<1
                    err(n,['Parameter <' name '> should be an integer number greater than 0. Check "' value '"']);return;
                end
                learning.config.maxBackwardSteps=v;
                learn(4)=1;
            case 'forwardStopCoefficient'
                v=str2double(value);
                if v<0 || v > 10^-1
                    err(n,['Parameter <' name '> should be a real number in the interval [0,e-1]. Check "' value '"']);return;
                end
                learning.config.forwardStopCoefficient=v;
                learn(5)=1;
            case 'backwardStopCoefficient'
                v=str2double(value);
                if v<0 || v > 10^-1
                    err(n,['Parameter <' name '> should be a real number in the interval [0,e-1]. Check "' value '"']);return;
                end
                learning.config.backwardStopCoefficient=v;
                learn(6)=1;
            case 'stepsForValidation'
                v=str2double(value);
                if  isnan(v) || v<1 || v >1000
                    err(n,['Parameter <' name '> should be a real number in the interval [1,1000]. Check "' value '"']);return;
                end
                learning.config.stepsForValidation=v;
                learn(7)=1;
            case 'maxStepsForValidation'
                v=str2double(value);
                if isnan(v) || v<1 || v >50
                    err(n,['Parameter <' name '> should be a real number in the interval [1,50]. Check "' value '"']);return;
                end
                learning.config.maxStepsForValidation=v;
                learn(8)=1;
            case 'deltaMax'
                v=str2double(value);
                if ~isempty(findstr(value,',')) || (~isempty(findstr(value,'.'))) || isnan(v) || v<1 || v >200
                    err(n,['Parameter <' name '> should be a real number in the interval [1,200]. Check "' value '"']);return;
                end
                learning.config.rProp.deltaMax=v;
                learn(9)=1;
            case 'deltaMin'
                v=str2double(value);
                if v<10^-10 || v > 10^-2
                    err(n,['Parameter <' name '> should be a real number in the interval (e-15,e-2]. Check "' value '"']);return;
                end
                learning.config.rProp.deltaMin=v;
                learn(10)=1;
            case 'etaP'
                v=str2double(value);
                if v<1.1 || v > 1.5
                    err(n,['Parameter <' name '> should be a real number in the interval [1.1,1.5]. Check "' value '"']);return;
                end
                learning.config.rProp.etaP=v;
                learn(11)=1;
            case 'etaM'
                v=str2double(value);
                if v<0.2 || v > 0.9
                    err(n,['Parameter <' name '> should be a real number in the interval [0.2,0.9]. Check "' value '"']);return;
                end
                learning.config.rProp.etaM=v;
                learn(12)=1;               
            case 'useValidation'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useValidation=0;
                else
                    if v==1
                        if ~isfield(dataSet,'validationSet')||isempty(dataSet.validationSet)
                            err(0,'You choose <useValidation>=1 but I can''t find any validationSet');return;
                        end
                        dynamicSystem.config.useValidation=1;
                    else
                        dynamicSystem.config.useValidation=0;
                    end
                end
                learn(13)=1;
            case 'useValidationMistakenPatterns'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useValidationMistakenPatterns=0;
                else
                    if v==1 && size(strfind(dataSet.config.type,'classification'),2)==0
                        warn(n, ['Parameter <useValidationMistakenPatterns>=1 but the the problem is not a classification one. I set <useValidationMistakenPatterns> to 0 for you']);
                        dynamicSystem.config.useValidationMistakenPatterns=0;
                    else
                        if v==1
                            dynamicSystem.config.useValidationMistakenPatterns=1;
                            learning.history.validationMistakenPatterns=[];
                            learning.current.bestValMistakenPatterns=intmax;
                        else
                            dynamicSystem.config.useValidationMistakenPatterns=0;
                        end
                    end
                end
                learn(14)=1;
            case 'saveErrorHistory'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.saveErrorHistory=0;
                else
                    dynamicSystem.config.saveErrorHistory=v;
                end
                learn(15)=1;
            case 'saveJacobianHistory'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.saveJacobianHistory=0;
                else
                    dynamicSystem.config.saveJacobianHistory=v;
                end
                learn(16)=1;
            case 'saveSaturationHistory'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.saveSaturationHistory=0;
                else
                    dynamicSystem.config.saveSaturationHistory=v;
                end
                learn(17)=1;
            case 'saveStabilityCoefficientHistory'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.saveStabilityCoefficientHistory=0;
                else
                    dynamicSystem.config.saveStabilityCoefficientHistory=v;
                end
                learn(18)=1;
            case 'saveIterationHistory'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.saveIterationHistory=0;
                else
                    dynamicSystem.config.saveIterationHistory=v;
                end
                learn(19)=1;
            case 'saveStateHistory'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.saveStateHistory=0;
                else
                    dynamicSystem.config.saveStateHistory=v;
                end
                learn(20)=1;
            otherwise
                err(n,['Unknown parameter <' name '> in Learning Parameters section']);return;
        end
    elseif section == 'g'
        switch name
            case 'useLogFile'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useLogFile=0;
                else
                    if v==1 dynamicSystem.config.useLogFile=1; else dynamicSystem.config.useLogFile=0; end
                end
                general(2)=1;
            case 'useAutoSave'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useAutoSave=0;
                else
                    if v==1 dynamicSystem.config.useAutoSave=1; else dynamicSystem.config.useAutoSave=0; end
                end
                general(3)=1;
            case 'useBalancedDataset'
                v=str2double(value);
                if  isnan(v) || (v~=0 && v~=1)
                    warn(0,['The parameter <' name '> is neither 0 or 1. I set it to 0 for you.'])
                    dynamicSystem.config.useBalancedDataset=0;
                else
                    
                    if v==1 && size(strfind(dataSet.config.type,'classification'),2)==0
                        warn(n, ['Parameter <useBalancedDataset>=1 but the the problem is not a classification one. I set <useBalancedDataset> to 0 for you']);
                        dynamicSystem.config.useBalancedDataset=0;
                    else
                        if v==1 dynamicSystem.config.useBalancedDataset=1; else dynamicSystem.config.useBalancedDataset=0; end
                    end
                end
                general(4)=1;
            otherwise
                err(n,['Unknown parameter <' name '> in General Parameters section']);
        end
    end
catch
    return;
end



% check if all section have been finished
function check_closed
global model learn general
if model(1)==0
    err(0,'I don''t find any Model Parameters section');return;
elseif model(1)==1
    err(0,'Model Parameters section has not been closed');return;
elseif learn(1)==0
    err(0,'I don''t find any Learning Parameters section');return;
elseif learn(1)==1
    err(0,'Learning Parameters section has not been closed');return;
elseif general(1)==0
    err(0,'I don''t find any General Parameters section');return;
elseif general(1)==1
    err(0,'General Parameters section has not been closed');return;
end



% Check if all necessary parameters has been set
function ok=check_thereisall
global model model_name learn learn_name general general_name
ok=1;
if model(2)==0
    err(0,'No model type in Model Parameters section');
elseif model(2)==1 %linear
    i=[3:15 18 19 21:23];
else
    i=[3:7 12:23];
end
k=find(model(i)==0);
for n=1:size(k,2)
    err(0,['No parameter <' char(model_name(i(k(n)))) '> in Model Parameters section']);
    ok=0;
end
y=find(learn==0);
for n=1:size(y,2)
    err(0,['No parameter <' char(learn_name(y(n))) '> in Learn Parameters section']);
    ok=0;
end
y=find(general==0);
for n=1:size(y,2)
    err(0,['No parameter <' char(general_name(y(n))) '> in General Parameters section']);
    ok=0;
end


%decrease dataSet size transforming some matrices in logical
function optimize
global dataSet dynamicSystem
dataSet.trainSet.connMatrix=logical(dataSet.trainSet.connMatrix);
if isfield(dataSet,'validationSet')
    dataSet.validationSet.connMatrix=logical(dataSet.validationSet.connMatrix);
end
if isfield(dataSet,'testSet')
    dataSet.testSet.connMatrix=logical(dataSet.testSet.connMatrix);
end
if dynamicSystem.config.useBalancedDataset
    balanceDataset;
else
    %in case that we've tried to balance the dataSet previously we've to set
    %again all elements to 1 before converting to logical
    dataSet.trainSet.maskMatrix=spones(dataSet.trainSet.maskMatrix);
    dataSet.trainSet.maskMatrix=logical(dataSet.trainSet.maskMatrix);

    if isfield(dataSet,'validationSet')
        dataSet.validationSet.maskMatrix=spones(dataSet.validationSet.maskMatrix);
        dataSet.validationSet.maskMatrix=logical(dataSet.validationSet.maskMatrix);
    end
end

