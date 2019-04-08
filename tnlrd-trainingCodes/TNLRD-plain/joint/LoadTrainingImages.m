function [clean, noisy] = LoadTrainingImages(R, path, sigma)

clean = [];

parfor pic_idx=1:R
    file = strcat(path,sprintf('test_%03d.png', pic_idx));
    img = double(imread(file));
    clean = [clean img(:)];
end

randn('seed', 0);
noisy = clean + sigma*randn(size(clean));

end

