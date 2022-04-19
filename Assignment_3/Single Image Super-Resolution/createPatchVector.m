function [vec_patches]= createPatchVector(img,boundary)
counter=1;
dim=size(img);
global size_patch;
num_patches = (dim(1)-2*boundary)*(dim(2)-2*boundary);
vec_patches = zeros(size_patch,num_patches);
 
%creating vector_patch matrix to store all the patches of img

for m = boundary+1:dim(1)-boundary
    for n = boundary+1:dim(2)-boundary
            current_patch = img(m-boundary:m+boundary,n-boundary:n+boundary);
            vec_patches(:,counter) = current_patch(:)-mean(current_patch(:));%%mean subtracted
            counter=counter+1;

% to get back a patch corresponding to col_num, simply do:
% reshape(vec_patches(col_num,:),length_patch,length_patch)

% to get center of patch corresponding to col_num,
% A = col_num - 1
% B = dim(2)-2*boundary
% m = 1 + boundary + floor(A/B), n = 1 + boundary + rem(A,B)

    end
end

