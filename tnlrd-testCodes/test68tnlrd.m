clear all; 
close all;
% clc;

load JointTraining_7x7x5_7x7_400_stage=5_sigma=25.mat

%% MFs means and precisions
trained_model = save_trained_NonLocalmodel(TrainedModels);

%% model parameters
fsz       = TrainedModels.fsz;
fnlsz     = TrainedModels.fnlsz;
filtN     = TrainedModels.filtN;
bsz       = fsz + 1;
fsz3D     = TrainedModels.fsz3D;
wsz       = TrainedModels.wsz;
bd        = TrainedModels.bd;
run_stage = TrainedModels.stage;

%% pad and crop operation
bndry     = [bsz,bsz];
pad       = @(x) padarray(x,bndry,'symmetric','both');
crop      = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));

rt=zeros(68,1);
ps=zeros(68,1);
ssim=zeros(68,1);
fn = './68imgs/test';
for i=1:68
    tic;
    %% load image, add noise and pad boundaries
    sigma     = TrainedModels.sigma;
    path      = sprintf('%s%03d.png',fn,i);
    I0        = double(imread(path));
    randn('seed', 0);
    Im        = I0 + sigma*randn(size(I0));

    [R,C]     = size(I0);
    rms1      = sqrt(mean((Im(:) - I0(:)).^2));

    input     = pad(Im);
    noisy     = pad(Im);
    clean     = I0;
    [ne,~,we] = NLMatching(input, fsz3D, wsz, bd);
    ne        = ne+1;
    we        = ones(size(ne));
    %% run denoising, 5 stages
    for s = 1:run_stage
        deImg = denoisingOneStepNonLocalGMixMFs(noisy, input, trained_model{s}, ne, we);
        t     = crop(deImg);
        deImg = pad(t);
        input = deImg;
    end
    Tim = toc;
    x_star = t(:);
    %% recover image
    rms2             = sqrt(mean((x_star - I0(:)).^2));
    ps(i)            = 20*log10(255/rms2);
    recover          = reshape(x_star,R,C);
    [ssim(i),~]      = ssim_index(uint8(recover*255), uint8(I0));
    rt(i)            = Tim;
    %imwrite(uint8(recover),[num2str(i) '.png']);
end