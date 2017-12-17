function plotComparisonNetTrainingResults

global comparisonNetLearning
figure('Name','Learning results of the comparison Net','NumberTitle','off');, 
hold on
plot(1:size(comparisonNetLearning.history.trainErrorHistory,2),comparisonNetLearning.history.trainErrorHistory,'b');

plot([comparisonNetLearning.stepsForValidation:comparisonNetLearning.stepsForValidation:...
comparisonNetLearning.stepsForValidation*(size(comparisonNetLearning.history.validationErrorHistory,2)-1)],...
comparisonNetLearning.history.validationErrorHistory(2:end),'r');

legend('Training error', 'Validation error');
hold off