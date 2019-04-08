clear all; close all;
% clc;
addpath('lbfgs');

global TP

%% model parameters
TP.fsz    = 7;
TP.fnlsz  = 5;
TP.filtN  = 48;  % filter_size^2-1
TP.fsz3D  = [3 3 (TP.fnlsz - 1)/2];
TP.wsz    = 15;
TP.bd     = 30;
basis     = gen_dct2(TP.fsz);
TP.basis  = basis(:,2:end);
TP.nbasis = eye(TP.fnlsz,TP.fnlsz);
TP.stage  = 5;

%% load input images
R         = 400;
sigma     = 25;
path      = 'the filepath of FoETrainingSets180';

[TP.G_TRUTH, TP.F_NOISE] = LoadTrainingImages(R, path, sigma);

%% pad input images
[TP.PT, TP.T, TP.F_NOISE, TP.NEIG, TP.WEI] = PadNoisyPatches(TP, R);

%% training function
fn_foe    = @loss_with_gradient_unit_filters_LUT;
opts_foe  = lbfgs_options('iprint', -1, 'maxits', 200, 'factr', 1e9, 'cb', @test_callback,'m',5,'pgtol', 1e-3);

%% training intialize
[x0, TP.MFS] = Equal_Initialization(TP);
TP.INPUT     = TP.F_NOISE;

%% training
TP.iter = 0;

nfoe    = length(x0(:,1));
cof     = zeros(nfoe,TP.stage);
stage_err = zeros(TP.stage,1);
ndims   = size(TP.F_NOISE,2);
r       = sqrt(ndims);
c       = r;
bsz     = TP.fsz + 1;

for s = 1:TP.stage
    [x,fx,exitflag,userdata] = lbfgs(fn_foe,x0(:,s),opts_foe);
    cof(:,s) = x;
    
    %% do one gradient descent step
    [deImg, ~]    = loss_gradient_greedy(x, TP);
    loss          = sum(sum((TP.T*deImg - TP.G_TRUTH).^2))/R; 
    deImg         = TP.PT*deImg;
    TP.INPUT      = deImg;
    
    stage_err(s)  = loss;
    fprintf('stage: %d, training loss = %.6f\n', s, loss);
    
    %% save trained models
    TrainedModels.cof     = cof;                  % coeff vectors
    TrainedModels.fsz     = TP.fsz;               % size of spatial filter
    TrainedModels.fnlsz   = TP.fnlsz;             % number of non-local similar patches
    TrainedModels.filtN   = TP.filtN;             % number of spatial filters (fsz^2-1)
    TrainedModels.fsz3D   = TP.fsz3D;             % half of [fsz fsz fnlsz]
    TrainedModels.wsz     = TP.wsz;               % half size of the non-local search window
    TrainedModels.bd      = TP.bd;                % bandwidth
    TrainedModels.basis   = TP.basis;             % DCT 2-D spatial filter basis
    TrainedModels.nbasis  = TP.nbasis;            % DCT 1-D nonlocal filter basis
    TrainedModels.stage   = TP.stage;             % number of diffusion steps
    TrainedModels.MFS     = TP.MFS;

    TrainedModels.R       = R;                    % number of training samples
    TrainedModels.sigma   = sigma;                % noise level
    TrainedModels.opt     = opts_foe;             % parameters of training process
    TrainedModels.trProc  = userdata;             % object values along iteration

    fn = sprintf('GreedyTraining_%dx%dx%d_%dx%d_%d_stage=%d_sigma=%d.mat',TP.fsz,TP.fsz,TP.fnlsz,TP.fsz3D(1)*2+1,TP.fsz3D(1)*2+1,R,TP.stage,sigma);
    save(fn,'TrainedModels');
end