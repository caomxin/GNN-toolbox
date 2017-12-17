function save_learning(experiment)

global dynamicSystem learning testing


dname=['experiments/' experiment.name '/' experiment.class ];
runlog=[dname '/exp_' experiment.run_prefix '.mat'];

if exist(runlog)
	load(runlog, 'explog')
else
	if ~exist(dname)
		if ~exist(['experiments/' experiment.name] )
			mkdir (['experiments/' experiment.name ])
		end
		mkdir (dname)
	end
	explog={};
	explog.next=1;
	explog.best=1;
	explog.best_err=9999999;
	explog.experiment=experiment;
end

fname=[dname '/exp_' experiment.run_prefix '_' int2str(explog.next)];

curr_idx=explog.next;

if isfield( dynamicSystem, 'savedTo' )
	fname=dynamicSystem.savedTo;
	curr_idx=dynamicSystem.savedToIdx;
else
	dynamicSystem.savedTo=fname;
	dynamicSystem.savedToIdx=explog.next;
	explog.next=explog.next+1;
end

if (explog.best_err>learning.current.bestErrorOnValidation) 
	explog.best_err=learning.current.bestErrorOnValidation;
	explog.best=curr_idx;
end

disp(['saving to: -  ' fname ' - '])
save(fname, 'dynamicSystem','learning','testing');
save(runlog, 'explog');

