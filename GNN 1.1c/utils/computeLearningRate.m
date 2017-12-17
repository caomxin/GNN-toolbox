function [lr,rollBack,state]=computeLearningRate(step,lr,error,state)

%% This fucntion compute the next learning rate by using an adaptive procedure
%% It returns
%% lr - the learning rate
%% rollBack - a variable which is set to 1 when the parameters should be
%%            restored to the previous values. It is used for signaling that error was increased to much
%% state - a structure containing some variables which will be useful at
%% the next activations

action=9;
rollBack=0;
sle=size(state.lastErrors);
nLastErrors=sle(2);


%% If error was encreased then learning factor is too large: we decrease it
%% and we set roolback to 1. Moreover, the set of last error was flushed
%% action contains a code that describe the action carried out on the
%% learning factor. The variable is used for debugging/logging puroposes
if (nLastErrors>1 && error >state.lastErrors(1));
    lr=lr*0.60;
    state.action(step)=-100;
    rollBack=1;
    state.increasedLR=0;
    state.lastErrors=[];
    return
end

%% if the vector lastErrors does not still contain two error then we
%% cannot use the adptive learning. First we check that condition.
%% Then, if error was not increased then we look at the error at times i-2 and i-1 and the previous action
%% in order to decide what to do.

if(nLastErrors==2)

    %% learning factor was increased and learning became faster  ->
    %% we increase learning factor
    if ((state.lastErrors(1)-error) >= (state.lastErrors(2)-state.lastErrors(1))) && state.increasedLR==1
        lr=lr*1.1;
        state.action(step)=2;
        state.increasedLR=1;
        %% learning factor was increased and learning became slower  ->
        %% we decrease learning factor
    elseif ((state.lastErrors(1)-error) < (state.lastErrors(2)-state.lastErrors(1))) && state.increasedLR==1
        lr=lr/1.1;
        state.action(step)=-1;
        state.increasedLR=-1;
        %% learning factor was decreased and learning became faster  ->
        %% we decrease learning factor
    elseif ((state.lastErrors(1)-error) > (state.lastErrors(2)-state.lastErrors(1)))&& state.increasedLR==-1
        lr=lr/1.1;
        state.action(step)=-2;
        state.increasedLR=-1;
        %% learning factor was decreased and learning became slower  ->
        %% we increase learning factor
    elseif ((state.lastErrors(1)-error) <= (state.lastErrors(2)-state.lastErrors(1))) && state.increasedLR==-1
        lr=lr*1.1;
        state.action(step)=1;
        state.increasedLR=1;
    elseif state.increasedLR==0
        state.increasedLR=1;
        state.action(step)=10;

    end
end

%% Finally, we update the last errors vector
if(nLastErrors<2)
    state.lastErrors(nLastErrors+1)=0;
    state.action(step)=0;
end
state.lastErrors=circshift(state.lastErrors',1)';
state.lastErrors(1)=error;