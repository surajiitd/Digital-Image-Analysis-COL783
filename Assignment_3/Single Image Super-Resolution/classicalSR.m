function [Weight_mat weighted_image]=classicalSR(sigma_PSF,Y,up_factor,boundary,scale_change,sigma_sq)

t=cputime;
[vec_patches]= createPatchVector(Y,boundary);
display(['time taken to construct patch matrix: ' num2str(cputime-t)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% nearest neighbour search
nn=9;%number of nearest neighbors
eps=1;%epsilon for ANN
t=cputime;

[x,y]=meshgrid(-boundary:boundary,-boundary:boundary);
gaussian_window=exp(-(x+y).^2/(4*sigma_PSF^2*(scale_change)));


anno=ann(vec_patches.*repmat(gaussian_window(:),1,size(vec_patches,2)));
[idx dst] = ksearch(anno, vec_patches.*repmat(gaussian_window(:),1,size(vec_patches,2)), nn, eps, false);
close(anno);
display(['time taken to calculate nearest neignbor at all pixels: ' num2str(cputime-t)]);
clear vec_patches x y;


%% do subpixel alignments for the nearest neighbor patches found

[sub_pixel_alignment_mat_row,sub_pixel_alignment_mat_col]=doSubpixel_alignment(idx,boundary,Y,up_factor);
clear gaussian_window;
%% Creating weighted_mat, and weighted image using callsical SR method
max_row_shift=max(sub_pixel_alignment_mat_row(:));
min_row_shift=min(sub_pixel_alignment_mat_row(:));

max_col_shift=max(sub_pixel_alignment_mat_col(:));
min_col_shift=min(sub_pixel_alignment_mat_col(:));

zz=floor(boundary);
[x,y]=meshgrid(-zz:zz,-zz:zz);

gaussian_window=exp(-(x+y).^2/(2*sigma_PSF^2*scale_change));

[z1 z2]=size(Y);
B = z2-2*boundary;
Y=double(Y);

%Creating equivalent LR{i,j} images by weighing patches shifted by (i,j) by
%the exp(-dst/sigma^2)
%Weight{i,j}: corresponding weight matrix, pixel wise total weight(as fn. of dst)
%Num_patches_pxl{i,j} : Number of patches at a pixel in LR{i,j}

LR=cell(max_row_shift-min_row_shift+1,max_col_shift-min_col_shift+1);
Weight=cell(max_row_shift-min_row_shift+1,max_col_shift-min_col_shift+1);
%Num_patches_pxl=cell(max_row_shift-min_row_shift+1,max_col_shift-min_col_shift+1);

%iterating over (i,j) values to create above matrices
for i=min_row_shift:max_row_shift
    for j=min_col_shift:max_col_shift
        row_shift=i-min_row_shift+1;
        col_shift=j-min_col_shift+1;        
        
        LR{row_shift,col_shift}=zeros(size(Y));
        Weight{row_shift,col_shift}=zeros(size(Y));
        %Num_patches_pxl{row_shift,col_shift}=zeros(size(Y));%10^-7 added to avoid divide by 0 error
    end
end


temp_size=size(idx);
num_patches=temp_size(2);
nn=temp_size(1);
        
for m=1:num_patches
    for n=1:nn
        row_shift=sub_pixel_alignment_mat_row(n,m);
        col_shift=sub_pixel_alignment_mat_col(n,m);
        i=row_shift-min_row_shift+1;
        j=col_shift-min_col_shift+1;        
        
        
        A = m - 1;
        p_org = 1 + boundary + floor(A/B); q_org = 1 + boundary + rem(A,B);
        patch_original=Y(p_org-boundary:p_org+boundary,q_org-boundary:q_org+boundary);
        A = double(idx(n,m)) - 1;
        p = 1 + boundary + floor(A/B); q = 1 + boundary + rem(A,B);
        dist=double(dst(n,m));
        
        window=gaussian_window*exp(-dist/sigma_sq(m));%(1/(dist+1))
        Weight{i,j}(p_org-zz:p_org+zz,q_org-zz:q_org+zz)=...
            Weight{i,j}(p_org-zz:p_org+zz,q_org-zz:q_org+zz)+ones(2*zz+1,2*zz+1).*window;
        %Num_patches_pxl{i,j}(p_org-boundary:p_org+boundary,q_org-boundary:q_org+boundary)=...
            %Num_patches_pxl{i,j}(p_org-boundary:p_org+boundary,q_org-boundary:q_org+boundary)+ones(2*boundary+1,2*boundary+1);
        similar_patch=Y(p-zz:p+zz,q-zz:q+zz)-mean2((Y(p-zz:p+zz,q-zz:q+zz)))+mean2(Y(p_org-zz:p_org+zz,q_org-zz:q_org+zz));
        LR{i,j}(p_org-zz:p_org+zz,q_org-zz:q_org+zz)=...
            LR{i,j}(p_org-zz:p_org+zz,q_org-zz:q_org+zz)+similar_patch.*window;
        

    end
end

%initialize HR to zeros
HR = zeros(round(up_factor*z1),round(up_factor*z2));%don't go on the name of this variable
Weight_mat=zeros(round(up_factor*z1),round(up_factor*z2));
Num_patches=zeros(round(up_factor*z1),round(up_factor*z2));

for i=min_row_shift:max_row_shift
    for j=min_col_shift:max_col_shift
        
        if  (abs(i)<(2*boundary+1)*up_factor/2) && (abs(j)<(2*boundary+1)*up_factor/2)% &&(i~=0 && j~=0)
            idx_i=i-min_row_shift+1;
            idx_j=j-min_col_shift+1;

            %if (i~=0 && j~=0)
            HR=circshift(HR, [i, j]);
            Weight_mat=circshift(Weight_mat, [i, j]);
            Num_patches=circshift(Num_patches, [i, j]);

            HR(1:up_factor:end, 1:up_factor:end)=...
                HR(1:up_factor:end, 1:up_factor:end)+LR{idx_i,idx_j};

            %Num_patches(1:up_factor:end, 1:up_factor:end)=...
             %   Num_patches(1:up_factor:end, 1:up_factor:end)+Num_patches_pxl{idx_i,idx_j};

            Weight_mat(1:up_factor:end, 1:up_factor:end)=...
                Weight_mat(1:up_factor:end, 1:up_factor:end)+Weight{idx_i,idx_j};

            HR=circshift(HR, -[i, j]);
            Num_patches=circshift(Num_patches, -[i, j]);
            Weight_mat=circshift(Weight_mat, -[i, j]);
        end
    end
end

%Accounting for the original patches in original Y
%constraints from original Low resolution image
 HR(1:up_factor:end, 1:up_factor:end)=...
             HR(1:up_factor:end, 1:up_factor:end)+Y;
 Weight_mat(1:up_factor:end, 1:up_factor:end)=...
             Weight_mat(1:up_factor:end, 1:up_factor:end)+1;
 %Num_patches(1:up_factor:end, 1:up_factor:end)=...
    %Num_patches(1:up_factor:end, 1:up_factor:end)+1;



%[x y]=meshgrid(1:size(Y,2),1:size(Y,1));
%[xi yi]=meshgrid(1:1/up_factor:size(Y,2),1:1/up_factor:size(Y,1));
%img=zeros(round(up_factor*z1),round(up_factor*z2));
weighted_image=zeros(round(up_factor*z1),round(up_factor*z2));

% zzz=interp2(x,y,Y,xi,yi,'cubic');
% temp_img=double(imresize_old(Y,up_factor,'bicubic'));
% img(end-up_factor+1:end,:)=temp_img(end-up_factor+1:end,:);
% img(:,end-up_factor+1:end)=temp_img(:,end-up_factor+1:end);
% % 
% % 
% clear temp_img;
% img(1:size(zzz,1),1:size(zzz,2))=zzz;
% clear zzz;

weighted_image=zeros(size(HR));
Weight_mat((isnan(Weight_mat)))=0;
weighted_image((Weight_mat~=0))=HR((Weight_mat~=0))./Weight_mat((Weight_mat~=0));
 
 
%weighted_image((Weight_mat==0))=img((Weight_mat==0));



