function test_learning(experiment)

global dynamicSystem learning testing dataSet


dname=['experiments/' experiment.name '/' experiment.class ];
runlog=[dname '/exp_' experiment.run_prefix '.mat'];

if exist(runlog)
	load(runlog, 'explog')
else
	err(0, 'experiment log not found');
end

accuracies.validation=[];
accuracies.test=[];
accuracies.train=[];

for i=[1:explog.next-1]

	fname=[dname '/exp_' experiment.run_prefix '_' int2str(i)];
	
	disp(sprintf('\n\n******'))
	disp(['** Experiment >>' int2str(i) '<<'])
	disp(['**   file: -  ' fname ' - '])
	
	
	load(fname);
	test
	accuracies.validation(end+1)=testing.optimal.validationSet.accuracy;
	accuracies.test(end+1)=testing.optimal.testSet.accuracy;
	accuracies.train(end+1)=testing.optimal.trainSet.accuracy;
	
end

header='';
msgVa='';
msgTr='';
msgTe='';

for i=[1:explog.next-1]
	header=sprintf('%s%21s', header, ['exp_' experiment.run_prefix '_' num2str(i)]); 
	msgVa=sprintf('%s%20.2s%%', msgVa, num2str(accuracies.validation(i)*100));
	msgTr=sprintf('%s%20.2s%%', msgTr, num2str(accuracies.train(i)*100));
	msgTe=sprintf('%s%20.2s%%', msgTe, num2str(accuracies.test(i)*100));
end

disp(header);
disp(msgVa);
disp(msgTr);
disp(msgTe);


best.validation=find(accuracies.validation==max(accuracies.validation));
best.test=find(accuracies.test==max(accuracies.test));
best.train=find(accuracies.train==max(accuracies.train));

disp(['The best performing model on validation:' num2str(best.validation)] );
disp(['The best performing model on train:' num2str(best.train)] );
disp(['The best performing model on test:' num2str(best.test)] );

