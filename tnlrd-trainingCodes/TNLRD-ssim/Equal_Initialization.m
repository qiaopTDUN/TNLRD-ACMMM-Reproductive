function [x0, mfs] = Equal_Initialization(TP)
load w0_63_means.mat;

mfs.means     = means;
mfs.precision = precision;
mfs.NumW      = length(means);
step          = 0.2;
delta         = 10;
D             = -delta+means(1):step:means(end)+delta;
D_mu          = bsxfun(@minus, D, means(:));
mfs.step      = step;
mfs.D         = D;
mfs.D_mu      = D_mu;
mfs.offsetD   = D(1);
mfs.nD        = numel(D);
mfs.G         = exp(-0.5*mfs.precision*D_mu.^2);

%% intialize using greedy training models
% load training_5x5_400_180_NoBeta_PSNR.mat;
% load training_7x7_400_180_NoBeta.mat;
% load training_9x9_400_180_NoBeta.mat;
% load GreedyTraining_7x7_400_180x180_sigma=15.mat;
% x0 = cof(:,1:run_stage);

%% initalize using plain intialization
filtN     = TP.filtN;
fnlsz     = TP.fnlsz;
run_stage = TP.stage;

w         = repmat(w,[1,filtN]);
cof_beta  = eye(filtN,filtN);
theta     = [10 5 ones(1,run_stage-2)];
p         = [log(1) log(0.1)*ones(1,run_stage-1)];

x0 = zeros(length(cof_beta(:)) + 1 + filtN*mfs.NumW + fnlsz*filtN, run_stage);
for i=1:run_stage
    tw      = zeros(fnlsz,filtN);
    tw(1,:) = 1;
    x0(:,i) = [cof_beta(:); p(i); w(:)*theta(i); tw(:)];
end