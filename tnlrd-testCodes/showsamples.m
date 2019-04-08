function showsamples(n)
global F_NOISE G_TRUTH INPUT T
R = size(F_NOISE,2);
indperm = randperm(R);
idx = indperm(1:n);

noisy = T*F_NOISE(:,idx);
clean = G_TRUTH(:,idx);
recover = T*INPUT(:,idx);
patch_size = sqrt(size(G_TRUTH,1));
sfigure(1);
displayDictionaryElementsAsImage([clean noisy recover],3,n,patch_size,patch_size,0);
drawnow;