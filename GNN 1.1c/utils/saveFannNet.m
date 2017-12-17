function saveFannNet(file)

global learning dynamicSystem


optimalParam = learning.current.optimalParameters;
sysconfig = dynamicSystem.config;
learnconfig = learning.config;
a = pwd;
cd /tmp

if nargin == 0
    file='/tmp/optimalFann';
end

%%%% transition net %%%%
h=fopen([file '_transition'],'wt');
if h==-1
    error(['Cannot open file ' file '_transition']);
else
    nInputs=dynamicSystem.config.transitionNet.nInputs;
    nHiddens=dynamicSystem.config.transitionNet.nHiddens;
    nOuts=dynamicSystem.config.transitionNet.nOuts;
    fprintf(h,'FANN_FLO_2.1\n');
    fprintf(h,'num_layers=3\n');
    fprintf(h,'learning_rate=0.700000\n');
    fprintf(h,'connection_rate=1.000000\n');
    fprintf(h,'network_type=0\n');
    fprintf(h,'learning_momentum=0.000000\n');
    fprintf(h,'training_algorithm=2\n');
    fprintf(h,'train_error_function=1\n');
    fprintf(h,'train_stop_function=0\n');
    fprintf(h,'cascade_output_change_fraction=0.010000\n');
    fprintf(h,'quickprop_decay=-0.000100\n');
    fprintf(h,'quickprop_mu=1.750000\n');
    fprintf(h,'rprop_increase_factor=1.200000\n');
    fprintf(h,'rprop_decrease_factor=0.500000\n');
    fprintf(h,'rprop_delta_min=0.000001\n');
    fprintf(h,'rprop_delta_max=1.000000\n');
    fprintf(h,'rprop_delta_zero=0.500000\n');
    fprintf(h,'cascade_output_stagnation_epochs=12\n');
    fprintf(h,'cascade_candidate_change_fraction=0.010000\n');
    fprintf(h,'cascade_candidate_stagnation_epochs=12\n');
    fprintf(h,'cascade_max_out_epochs=150\n');
    fprintf(h,'cascade_max_cand_epochs=150\n');
    fprintf(h,'cascade_num_candidate_groups=2\n');
    fprintf(h,'bit_fail_limit=4.00000005960464477539e-01\n');
    fprintf(h,'cascade_candidate_limit=1.00000000000000000000e+03\n');
    fprintf(h,'cascade_weight_multiplier=4.00000005960464477539e-01\n');
    fprintf(h,'cascade_activation_functions_count=8\n');
    fprintf(h,'cascade_activation_functions=3 5 7 8 10 11 14 15\n');
    fprintf(h,'cascade_activation_steepnesses_count=4\n');
    fprintf(h,'cascade_activation_steepnesses=2.50000000000000000000e-01 5.00000000000000000000e-01 7.50000000000000000000e-01 1.00000000000000000000e+00\n');
    fprintf(h,['layer_sizes=' num2str(nInputs+1) ' ' num2str(nHiddens+1) ' ' num2str(nOuts+1) '\n']);  %+1 because of the bias
    fprintf(h,'scale_included=0\n');
    
    % neurons
    fprintf(h,'neurons (num_inputs, activation_function, activation_steepness)=');
    for i=1:nInputs
        fprintf(h,'(0, 0, 0.00000000000000000000e+00) ');
    end
    fprintf(h,'(0, 0, 0.00000000000000000000e+00) '); % bias
    for i=1:nHiddens
        fprintf(h,['(' num2str(nInputs+1) ', 3, 5.00000000000000000000e-01) ']); % activation sigmoid
    end
    fprintf(h,'(0, 3, 0.00000000000000000000e+00) '); % bias
    for i=1:nOuts
        fprintf(h,['(' num2str(nHiddens+1) ', 3, 5.00000000000000000000e-01) ']); % activation sigmoid
    end
    fprintf(h,'(0, 3, 0.00000000000000000000e+00)\n'); % bias
    
    % connections
    fprintf(h,'connections (connected_to_neuron, weight)=');
    w1=optimalParam.transitionNet.weights1;
    b1=optimalParam.transitionNet.bias1;
    for hid=1:size(w1,1)
        for inp=1:size(w1,2)
            fprintf(h,'(%d, %21.20e) ', [inp-1 w1(hid,inp)]);
        end
        fprintf(h,'(%d, %21.20e) ', [inp b1(hid)]);
    end
    w2=optimalParam.transitionNet.weights2;
    b2=optimalParam.transitionNet.bias2;
    for out=1:size(w2,1)
        for hid=1:size(w2,2)
            fprintf(h,'(%d, %21.20e) ', [nInputs+hid w2(out,hid)]);
        end
        fprintf(h,'(%d, %21.20e) ', [nInputs+hid+1 b2(out)]);
    end
    fprintf(h,'\n\n');  
end


%%%% out net %%%%
h=fopen([file '_out'],'wt');
if h==-1
    error(['Cannot open file ' file '_out']);
else
    nInputs=dynamicSystem.config.outNet.nInputs;
    nHiddens=dynamicSystem.config.outNet.nHiddens;
    nOuts=dynamicSystem.config.outNet.nOuts;
    fprintf(h,'FANN_FLO_2.1\n');
    fprintf(h,'num_layers=3\n');
    fprintf(h,'learning_rate=0.700000\n');
    fprintf(h,'connection_rate=1.000000\n');
    fprintf(h,'network_type=0\n');
    fprintf(h,'learning_momentum=0.000000\n');
    fprintf(h,'training_algorithm=2\n');
    fprintf(h,'train_error_function=1\n');
    fprintf(h,'train_stop_function=0\n');
    fprintf(h,'cascade_output_change_fraction=0.010000\n');
    fprintf(h,'quickprop_decay=-0.000100\n');
    fprintf(h,'quickprop_mu=1.750000\n');
    fprintf(h,'rprop_increase_factor=1.200000\n');
    fprintf(h,'rprop_decrease_factor=0.500000\n');
    fprintf(h,'rprop_delta_min=0.000001\n');
    fprintf(h,'rprop_delta_max=1.000000\n');
    fprintf(h,'rprop_delta_zero=0.500000\n');
    fprintf(h,'cascade_output_stagnation_epochs=12\n');
    fprintf(h,'cascade_candidate_change_fraction=0.010000\n');
    fprintf(h,'cascade_candidate_stagnation_epochs=12\n');
    fprintf(h,'cascade_max_out_epochs=150\n');
    fprintf(h,'cascade_max_cand_epochs=150\n');
    fprintf(h,'cascade_num_candidate_groups=2\n');
    fprintf(h,'bit_fail_limit=4.00000005960464477539e-01\n');
    fprintf(h,'cascade_candidate_limit=1.00000000000000000000e+03\n');
    fprintf(h,'cascade_weight_multiplier=4.00000005960464477539e-01\n');
    fprintf(h,'cascade_activation_functions_count=8\n');
    fprintf(h,'cascade_activation_functions=3 5 7 8 10 11 14 15\n');
    fprintf(h,'cascade_activation_steepnesses_count=4\n');
    fprintf(h,'cascade_activation_steepnesses=2.50000000000000000000e-01 5.00000000000000000000e-01 7.50000000000000000000e-01 1.00000000000000000000e+00\n');
    fprintf(h,['layer_sizes=' num2str(nInputs+1) ' ' num2str(nHiddens+1) ' ' num2str(nOuts+1) '\n']);  %+1 because of the bias
    fprintf(h,'scale_included=0\n');
    
    % neurons
    fprintf(h,'neurons (num_inputs, activation_function, activation_steepness)=');
    for i=1:nInputs
        fprintf(h,'(0, 0, 0.00000000000000000000e+00) ');
    end
    fprintf(h,'(0, 0, 0.00000000000000000000e+00) '); % bias
    for i=1:nHiddens
        fprintf(h,['(' num2str(nInputs+1) ', 3, 5.00000000000000000000e-01) ']); % activation sigmoid
    end
    fprintf(h,'(0, 3, 0.00000000000000000000e+00) '); % bias
    for i=1:nOuts
        fprintf(h,['(' num2str(nHiddens+1) ', 3, 5.00000000000000000000e-01) ']); % activation sigmoid
    end
    fprintf(h,'(0, 3, 0.00000000000000000000e+00)\n'); % bias
    
    % connections
    fprintf(h,'connections (connected_to_neuron, weight)=');
    w1=optimalParam.outNet.weights1;
    b1=optimalParam.outNet.bias1;
    for hid=1:size(w1,1)
        for inp=1:size(w1,2)
            fprintf(h,'(%d, %21.20e) ', [inp-1 w1(hid,inp)]);
        end
        fprintf(h,'(%d, %21.20e) ', [inp b1(hid)]);
    end
    w2=optimalParam.outNet.weights2;
    b2=optimalParam.outNet.bias2;
    for out=1:size(w2,1)
        for hid=1:size(w2,2)
            fprintf(h,'(%d, %21.20e) ', [nInputs+hid w2(out,hid)]);
        end
        fprintf(h,'(%d, %21.20e) ', [nInputs+hid+1 b2(out)]);
    end
    fprintf(h,'\n\n');  
end



cd(a)