function I = dictionary2image(D, nlK)
    [ksz, N] = size(D);
    ksz = sqrt(ksz);
    bsz = 40;
    nc = 12;
    nr = 4;
    scale = 10;
    
    I = ones((bsz+ksz*scale)*nr+bsz,(bsz+ksz*scale)*nc+bsz,3);
    I(:,:,1) = 1;
    I(:,:,2) = 1;
    I(:,:,3) = 1;
    
    k = 1;
    for i = 1:nc
        for j = 1:nr
            if k > N
                continue;
            end
            
            si = reshape(D(:,k),ksz,ksz);
            minD = min(si(:));
            maxD = max(si(:));
            si = (si - minD) / (maxD - minD);
            
            I((bsz+ksz*scale)*(j-1)+bsz:(ksz*scale+bsz)*(j-1)+ksz*scale+bsz+2,(bsz+ksz*scale)*(i-1)+bsz:(ksz*scale+bsz)*(i-1)+ksz*scale+bsz+2,:) = 0;
            for ii = 1:ksz
                for jj = 1:ksz
                    I((bsz+ksz*scale)*(j-1)+bsz+1+(jj-1)*scale:(ksz*scale+bsz)*(j-1)+jj*scale+bsz+1,(bsz+ksz*scale)*(i-1)+bsz+1+(ii-1)*scale:(ksz*scale+bsz)*(i-1)+ii*scale+bsz+1,1) = si(jj,ii);
                    I((bsz+ksz*scale)*(j-1)+bsz+1+(jj-1)*scale:(ksz*scale+bsz)*(j-1)+jj*scale+bsz+1,(bsz+ksz*scale)*(i-1)+bsz+1+(ii-1)*scale:(ksz*scale+bsz)*(i-1)+ii*scale+bsz+1,2) = si(jj,ii);
                    I((bsz+ksz*scale)*(j-1)+bsz+1+(jj-1)*scale:(ksz*scale+bsz)*(j-1)+jj*scale+bsz+1,(bsz+ksz*scale)*(i-1)+bsz+1+(ii-1)*scale:(ksz*scale+bsz)*(i-1)+ii*scale+bsz+1,3) = si(jj,ii);
                end
            end
            
            si = nlK(:,k);
            minD = min(si(:));
            maxD = max(si(:));
            si = (si - minD) / (maxD - minD);
            
            dshift = (bsz-scale)/2;
            nlsz = length(si);
            lshift = (ksz-nlsz)/2*scale;
            I((bsz+ksz*scale)*(j-1)+bsz+ksz*scale+dshift:(ksz*scale+bsz)*(j-1)+ksz*scale+bsz+1+dshift+scale,(bsz+ksz*scale)*(i-1)+bsz+lshift:(ksz*scale+bsz)*(i-1)+ksz*scale+bsz+1-lshift,:) = 0;
            for jj = 1:length(si)
                I((bsz+ksz*scale)*(j-1)+bsz+1+ksz*scale+dshift:(ksz*scale+bsz)*(j-1)+ksz*scale+bsz+dshift+scale,(bsz+ksz*scale)*(i-1)+bsz+1+(jj-1)*scale+lshift:(ksz*scale+bsz)*(i-1)+jj*scale+bsz+lshift,:) = si(jj);
            end
            
            k = k + 1;
        end
    end
    
    I = I(bsz-dshift+2:end-1, bsz-dshift+2:end-(bsz-dshift), :);
end