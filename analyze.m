% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019
clear; close all;

% initialize variables
names = {'Angle','Bump', 'CShape','GShape','JShape','JShape_2','Khamesh',...
    'Line','Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4',...
    'NShape','PShape','RShape','Saeghe','Sharpc','Sine','Soft_Sine',...
    'Spoon','Sshape','Trapezoid','WShape','Zshape'};
choseCLF = {'Sq','SOS','WSAQF'};%{'SOSsto'};%

path_data = './data/';
if contains(choseCLF{1},'sto')
    diaryfile = [path_data, 'SummarySto.txt'];
else
    diaryfile = [path_data, 'Summary.txt'];
end
if exist(diaryfile,'file'), delete(diaryfile); end
diary(diaryfile);


kk=0;
for nset = 1:size(names,2)
    set = names{nset};
    for  nCLF = 1:length(choseCLF)
        fname = [path_data,set,choseCLF{nCLF},'.mat'];
        if exist(fname,'file')
            if nCLF==1, kk = kk+1; available_sets{kk,1} = names{nset};end
            load(fname,'cv','CorrectionEffort','AreaError','t_learnCLF','E');
            if contains(choseCLF{nCLF},'sto')
                load(fname,'gprMdls');
                lmin = min(gprMdls{1}.KernelInformation.KernelParameters(1),gprMdls{2}.KernelInformation.KernelParameters(1));
                sigmax = max(gprMdls{1}.KernelInformation.KernelParameters(2),gprMdls{2}.KernelInformation.KernelParameters(2));
                isasym(kk,nCLF) = lmin^2 > E*sigmax^2;
                beta(kk,nCLF) = sqrt(lmin^2*lambertw(-E*sigmax^2/(lmin^2) * exp(-E*sigmax^2/(lmin^2))) + E*sigmax^2);
            else
                avAreaError(kk,nCLF) = sum(AreaError(cv==1))/sum(cv);
                avCorrectionEffort(kk,nCLF) = sum(CorrectionEffort(cv==1))/sum(cv);
            end
            conv(kk,nCLF) = sum(cv);
            T_learnCLF(kk,nCLF) = t_learnCLF;
        end
    end
    
end
if size(conv,2)>length(choseCLF),  choseCLF = [choseCLF {'SEDS'}];end

available_sets{end+1} = 'OVERALL(mean,sum)'; strout = [];


if contains(choseCLF{nCLF},'sto')
    isasym(end+1,:) = sum(isasym,1);beta(end+1,:) = sum(beta,1);
    disp('Asymptotic Stability?');
    disp(array2table(isasym,'RowNames', available_sets,'VariableNames',choseCLF));
    disp('Convergence radius');
    disp(array2table(beta,'RowNames', available_sets,'VariableNames',choseCLF));
else
    avAreaError(isinf(avAreaError)) = NaN; avAreaError(end+1,:) = nanmean(avAreaError,1);
    avCorrectionEffort(isinf(avCorrectionEffort)) = NaN; avCorrectionEffort(end+1,:) = nanmean(avCorrectionEffort,1);
    conv(end+1,:) = sum(conv,1);  T_learnCLF(end+1,:) = nanmean(T_learnCLF,1);
    
    disp('# of trajectories converged');
    disp(array2table(conv,'RowNames', available_sets,'VariableNames',choseCLF));
    disp('Area Error');
    disp(array2table(avAreaError,'RowNames', available_sets,'VariableNames',choseCLF));
    disp('Correction Effort');
    disp(array2table(avCorrectionEffort,'RowNames', available_sets,'VariableNames',choseCLF));
    disp('Time to learn CLF');
    disp(array2table(T_learnCLF,'RowNames', available_sets,'VariableNames',choseCLF));
end



diary off