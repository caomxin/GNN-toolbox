function start_learning(experiment)
global dataSet
load(['experiments/data/all_' experiment.class])
eval(['dataSet=all_' experiment.class ';'])
Configure
learn

