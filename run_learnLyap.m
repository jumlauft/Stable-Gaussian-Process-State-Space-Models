% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft 03/2019

clear; close all; clc; rng default
names = {'Angle','Bump','CShape','GShape','JShape','JShape_2','Khamesh',...% 1-7
    'Line','Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4',...% 8-12
    'NShape','PShape','RShape','Saeghe','Sharpc','Sine','Soft_Sine',... % 13 - 19
    'Spoon','Sshape','Trapezoid','WShape','Zshape'}; Nset = length(names); % 20 - 24
path_data='./data/';

% Include Estimation of Equilibrium point
% choseCLF = {'SOSEq'};     
% Stochastic Stimulation
% choseCLF = {'SOSsto'};    
% For comparison of different Lyaounov functions
choseCLF = {'Sq','SOS','WSAQF'}; 

ds = 1;             % Downsampling of training data (default = 1)
Shifttr = [0;0];     % Constant Shift of training data
sn = 1e-8*[1 1]';   % Observation noise (default = 1e-8)
optGPR = {'KernelFunction','ardsquaredexponential','ConstantSigma',true,'Sigma',sn,};

optLL.minP = 1e-5;  % Lower bound for EV of learned matrices (default = 1e-5)
optLL.maxP = 1e8;   % Upper bound for EV of learned matrices (default = 1e8)
optLL.dSOS = 2;     % Degree of Sum of Squares (default = 2)
optLL.dWSAQF = 3;   % Degree of WSAQF (default = 3)
optLL.opt = optimoptions('fmincon','Display','off','GradObj','off',...
    'CheckGradients',false,'MaxFunctionEvaluations',1e8,'MaxIterations',1e3,...
    'SpecifyConstraintGradient',false);

optCLF.rho = 0.02;    % Minimum Decrease of Lyapunov function
optCLF.minstep = 5e-1;% Minimum Step size for stabilized DS
optCLF.opt = optimoptions('fmincon','Display','off','GradObj','on',...
    'CheckGradients',false,'MaxFunctionEvaluations',1e8, 'MaxIterations',100,...
    'SpecifyConstraintGradient',false);

x0User=[-150 -120]'; % User-defined initial point (for stochastic case only)
Nrepx0 = 3;          % Repetition of simulation (for stochastic case only)

optSim.stopN = 1e3;  % Stopping condition simulation: # of steps (default = 1e3)
optSim.stopX = 5;    % Stopping condition simulation: proximity origin (default = 5)
optSim.xEq = Shifttr;


% Visualize
dovisualize = 0;    % Choose if each dataset is visualized
dss = 2;            % density of stream slice (default = 2)
dscont  = 15;       % Density of contour (default = 15)
Nte = 1e3;          % Number of test points (default = 1e3)
gmte = 20;          % Grid margin for test points beyond training points

for nset = 1:Nset% 9%13% 
    %% Load Training Data
    setname = names{nset}; demos = load(['./LASA/',setname, '.mat'],'demos');
    demos = demos.demos; Ndemo = length(demos); Xtr=[]; Ytr=[];%xtrain=[];
    for ndemo = 1:Ndemo
        Xtrtemp = demos{ndemo}(:,1:ds:end); Xtr = [Xtr Xtrtemp(:,1:end-1)];
        Ytr = [Ytr Xtrtemp(:,2:end)];   %xtrain(:,:,ndemo) = Xtrtemp;
    end
    Xtr = Xtr+Shifttr; Ytr = Ytr + Shifttr;
    dXtr = Ytr-Xtr; [E, Ntr] = size(Xtr);
    % Set test area
    Ndte = floor(nthroot(Nte,E)); % Nte = Ndte^E;
    grid_min = min(Xtr,[],2)-gmte;grid_max =max(Xtr,[],2)+gmte;
    Xte = ndgridj(grid_min,grid_max ,Ndte*ones(E,1)) ;
    Xte1 = reshape(Xte(1,:),Ndte,Ndte); Xte2 = reshape(Xte(2,:),Ndte,Ndte);
    
    %% Learn GPDM model
    disp([setname,': Training GP model...']);
    [mGP, varGP,gprMdls] = learnGPR(Xtr,Ytr,optGPR{:});
    
    
    %% Run Approach for each Lyapunov function
    for  nCLF=1:length(choseCLF)
        
        disp([setname, choseCLF{nCLF},': Learn CLF...'])
        tic;
        switch choseCLF{nCLF}
            case 'Vvar'
                V = Vvar(varGP,grid_min,grid_max,Ndte,zeros(E,1));xEq = zeros(E,1);
                Vclf = @(xi) V(xi(1,:)',xi(2,:)');
            case 'Sq'
                [val_learnCLF,P_Sq ] = learnSq(Xtr,Ytr,optLL);xEq = zeros(E,1);
                Vclf = @(x) Sq(x,P_Sq);
            case 'SqEq'
                [val_learnCLF,P_Sq,xEq] = learnSq(Xtr,Ytr,optLL);
                Vclf = @(x) Sq(x-xEq,P_Sq);
            case 'Sqsto'
                [val_learnCLF,P_Sq] = learnSq(Xtr,Ytr,optLL);xEq = zeros(E,1);
                Vclf = @(x) muSq(x,varGP(x),P_Sq);
            case 'SOS'
                [val_learnCLF,P_SOS ] = learnSOS(Xtr,Ytr,optLL);xEq = zeros(E,1);
                Vclf = @(x) SOS(x,P_SOS,optLL.dSOS);
            case 'SOSEq'
                [val_learnCLF,P_SOS,xEq] = learnSOS(Xtr,Ytr,optLL);
                Vclf = @(x) SOS(x-xEq,P_SOS,optLL.dSOS);
            case 'SOSsto'
                [val_learnCLF,P_SOS ] = learnSOS(Xtr,Ytr,optLL);xEq = zeros(E,1);
                Vclf = @(x) muSOS(x,varGP(x),P_SOS,optLL.dSOS);
            case 'WSAQF'
                [val_learnCLF,P0_WSAQF,Pl_WSAQF,mu_WSAQF] = learnWSAQF(Xtr,Ytr,optLL);xEq = zeros(E,1);
                Vclf = @(x) WSAQF(x,P0_WSAQF,Pl_WSAQF,mu_WSAQF);
            case 'WSAQFEq'
                [val_learnCLF,P0_WSAQF,Pl_WSAQF,mu_WSAQF,xEq] = learnWSAQF(Xtr,Ytr,optLL);
                Vclf = @(x) WSAQF(x-xEq,P0_WSAQF,Pl_WSAQF,mu_WSAQF);
            otherwise
                error('CLF not known');
        end
        dxdtfun = @(x) stableDS(x,GPSSMm,Vclf,rho);
        t_learnCLF = toc;
        %% Evaluate Test Points
        disp([setname, choseCLF{nCLF},': Evaluating Test Points...']);
        
        mGPte = mGP(Xte); varGPte = varGP(Xte);
        mGPstabte = CLF(Xte,mGPte,Vclf,optCLF);
        
        %% Simulate Trajectories
        disp([setname, choseCLF{nCLF},': Simulate Trajectories...']);
        x0s = cell2mat(cellfun(@(v) v(:,1), demos,'UniformOutput', false)) + Shifttr;
               
        if contains(choseCLF{nCLF},'sto')
            x0s =  repmat([x0s x0User],1,Nrepx0); optSim.stopN = max(optSim.stopN,5e2);
            [Xtraj_unst,CorrectionEffort_unst,cv_unst] = SimStableTraj(mGP,   @(x0,x1) x1,    x0s,optSim,varGP);
             [Xtraj,CorrectionEffort,cv] = SimStableTraj(mGP,  @(x0,x1)CLF(x0,x1,Vclf,optCLF), x0s,optSim,varGP);
             AreaError = NaN;
            name_vars = {'gprMdls','varGP'}; vars = {gprMdls,varGP};
        else
            [Xtraj_unst,CorrectionEffort_unst,cv_unst] = SimStableTraj(mGP,   @(x0,x1) x1,    x0s,optSim,varGP);
            [Xtraj,CorrectionEffort,cv] = SimStableTraj(mGP,  @(x0,x1)CLF(x0,x1,Vclf,optCLF), x0s, optSim);
            AreaError = errArea(Xtraj,demos);
            name_vars = {}; vars = {};
        end
        
        
        %% Plot and save
        disp([setname, choseCLF{nCLF},': Plot and Save...']);
        if dovisualize, visualize;  end
        name_vars = {name_vars{:},'optLL','ds','E','optCLF','optGPR','t_learnCLF','Shifttr','optSim','dss','dscont','Nte','gmte','setname','Xtr','Ytr','mGPte','varGPte','mGPstabte','dXtr','demos','Vclf','xEq','Xtraj','Xtraj_unst','cv','cv_unst','CorrectionEffort','AreaError'};
        vars =      {vars{:}     , optLL , ds , E , optCLF , optGPR , t_learnCLF , Shifttr , optSim , dss , dscont , Nte , gmte , setname , Xtr , Ytr, mGPte , varGPte , mGPstabte , dXtr , demos , Vclf , xEq , Xtraj , Xtraj_unst , cv , cv_unst , CorrectionEffort , AreaError};
        
         savej( [path_data,setname,choseCLF{nCLF}],name_vars,vars);
        
            
        disp([setname, choseCLF{nCLF},': Pau']);
    end
end
if nset == Nset,  analyze;end
