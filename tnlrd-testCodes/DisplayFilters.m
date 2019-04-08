function DisplayFilters(K,nRow,nCol)
for row = 1:nRow
    for col = 1:nCol
%         idx1 = (row-1)*nCol + col;
        idx2 = (col - 1)*nRow + row;
        subplot(nRow,nCol,idx2);
        
        tt = K{idx2};
        norm_filter = norm(tt(:));
        imshow(tt,[]);
        str1 = num2str(norm_filter,'%.2f');
        title(str1);
    end
end