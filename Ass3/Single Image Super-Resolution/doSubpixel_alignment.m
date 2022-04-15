function [sub_pixel_alignment_mat_row, sub_pixel_alignment_mat_col]=doSubpixel_alignment(idx,boundary,Y,up_factor)

 t=cputime;
%doing sub-pixel alignment
sub_pixel_alignment_mat_row=zeros(size(idx));
sub_pixel_alignment_mat_col=zeros(size(idx));


dim=size(Y);
B = dim(2)-2*boundary;
for i=1:size(idx,2)
        %storing the original patch in a    
        A = i - 1;
        m = 1 + boundary + floor(A/B); n = 1 + boundary + rem(A,B);
        %a=zeros(3*length_patch,3*length_patch);
        a=Y(m-boundary:m+boundary,n-boundary:n+boundary);
        %a=imresize(Y(m-boundary:m+boundary,n-boundary:n+boundary),up_factor,'bicubic');
        
    for j=1:size(idx,1)
        
        %storing the matched patch in b
        A = double(idx(j,i)) - 1;
        m = 1 + boundary + floor(A/B); n = 1 + boundary + rem(A,B);
        b=Y(m-boundary:m+boundary,n-boundary:n+boundary);
        [alignment]=dftregistration1(fft2(a),fft2(b),1);
        sub_pixel_alignment_mat_row(j,i)=round(up_factor*alignment(3));
        sub_pixel_alignment_mat_col(j,i)=round(up_factor*alignment(4));
        
    end
end

display(['Time in sub pixel alignment of all patches: ' num2str(cputime-t)]);
