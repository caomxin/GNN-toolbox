function plotTrainingResults
global learning

if isfield(learning.history, 'saturationCoefficient')
    figure('Name','Saturation Coefficients','NumberTitle','off');
    i=1;
    color=['b','r','k'];
    leg=cell(1,1);
    hold on
    for it = fieldnames(learning.history.saturationCoefficient)'
        plot([1:size(learning.history.saturationCoefficient.(char(it)),2)],learning.history.saturationCoefficient.(char(it)),color(i));
        leg{i} = char(it);
        i = i+1;
    end
    hold off;
    title('Saturation Coefficients');
    legend(char(leg));
end

if isfield(learning.history, 'forwardItHistory')
    figure('Name','Forward and backward iterations','NumberTitle','off');
    plot([1:size(learning.history.forwardItHistory,2)],learning.history.forwardItHistory,'b');
    hold on
    plot([1:size(learning.history.backwardItHistory,2)],learning.history.backwardItHistory,'g');
    hold off
    title('Forward and backward iterations');
    legend('Forward iterations', 'Backward iterations');
end

if isfield(learning.history, 'stabilityCoefficientHistory')
    figure('Name','Stability Coefficient History','NumberTitle','off');
    plot([1:size(learning.history.stabilityCoefficientHistory,2)],learning.history.stabilityCoefficientHistory);
    title('Stability Coefficient History');
end

if isfield(learning.history, 'jacobianHistory')
    figure('Name','Jacobian Error History','NumberTitle','off');
    plot([1:size(learning.history.jacobianHistory,2)],learning.history.jacobianHistory);
    title('Jacobian Error History');
end

if isfield(learning.history, 'jacobianHistoryComplete')
    figure('Name','Jacobian History','NumberTitle','off');
    plot([1:size(learning.history.jacobianHistoryComplete,2)],learning.history.jacobianHistoryComplete);
    title('Jacobian History');
end

if isfield(learning.history,'trainErrorHistory')
    figure('Name','Learning results','NumberTitle','off');
    plot([1:size(learning.history.trainErrorHistory,2)],learning.history.trainErrorHistory,'k');
    hold on
    t=[learning.config.stepsForValidation:learning.config.stepsForValidation:...
        learning.config.stepsForValidation*(size(learning.history.validationErrorHistory,2))];
    t(end)=learning.current.nSteps-1;
    plot(t, learning.history.validationErrorHistory,'r');
    hold off
    title('Learning results');
    legend('learning error', 'validation error');
end