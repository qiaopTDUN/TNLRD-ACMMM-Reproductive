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
fn_foe    = @loss_with_gradient_joint_stages;
opts_foe  = lbfgs_options('iprint', -1, 'maxits', 200, 'factr', 1e9, 'cb', @test_callback,'m',5,'pgtol', 1e-3);

%% training intialize
[x0, TP.MFS] = Equal_Initialization(TP);
load('JointTraining_7x7_400_180x180_stage=5_sigma=25.mat','cof');
x0(1:size(cof,1),:) = cof;

%% training
TP.iter = 0;
[x,fx,exitflag,userdata] = lbfgs(fn_foe,x0(:),opts_foe);

%% save trained models
TrainedModels.cof     = reshape(x,size(x0));  % coeff vectors
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

fn = sprintf('JointTraining_%dx%dx%d_%dx%d_%d_stage=%d_sigma=%d.mat',TP.fsz,TP.fsz,TP.fnlsz,TP.fsz3D(1)*2+1,TP.fsz3D(1)*2+1,R,TP.stage,sigma);
save(fn,'TrainedModels');