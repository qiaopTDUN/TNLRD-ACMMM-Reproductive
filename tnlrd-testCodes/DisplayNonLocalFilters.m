function DisplayNonLocalFilters(K,nlK,nRow,nCol)
for row = 1:nRow
    for col = 1:nCol
        idx1 = (col-1)*nRow + row;
        idx2 = 2*(row - 1)*nCol + col;
        subplot(2*nRow,nCol,idx2);
        
        tt = K{idx1};
        norm_filter = norm(tt(:));
        imshow(tt,[]);
        str1 = num2str(norm_filter,'%.2f');
        title(str1);
    end
    
    for col = 1:nCol
        idx1 = (col-1)*nRow + row;
        idx2 = (2*row-1)*nCol + col;
        subplot(2*nRow,nCol,idx2);
        tt = nlK(:,idx1);
        imshow(tt',[]);
    end
end