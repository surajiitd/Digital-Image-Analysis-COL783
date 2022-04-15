function [weight_mat_example I_m_example]=exampleSR(sigma_PSF,I,boundary,l,m,alpha,sigma_sq)

I_m_example=cell(m-l,1);%cell structure
weight_mat_example=cell(m-l,1);
%initialize the cell structure for the output image

Y=I{l};
dim=size(Y);
[z1,z2]=size(Y);    
[vec_patches_Y]=createPatchVector(Y,boundary);

%we construct I_m_example,weight_mat_example(cell structure) for each higher level image I(l+1)...I(m).

for scale=l+1:m
    fac=alpha^(scale-l);
    
    I_m_example{scale-l,1}=zeros(size(I{l})*fac);
    weight_mat_example{scale-l,1}=zeros(size(I{l})*fac);

    %create a patch vector for the LR images at level I{l-(scale-l)}
    img_LR=I{l-(scale-l)};
    
    [vec_patches_LR]=createPatchVector(img_LR,boundary);
    
    %% nearest neighbor search
    t=cputime;
    nn=9;
    eps=0.1;
    
    
    [x,y]=meshgrid(-boundary:boundary,-boundary:boundary);
    gaussian_window=exp(-(x+y).^2/(4*sigma_PSF^2*(scale-l)));
    
    %Gaussian SSD
    anno=ann(vec_patches_LR.*repmat(gaussian_window(:),1,size(vec_patches_LR,2)));
    [idx dst] = ksearch(anno, vec_patches_Y.*repmat(gaussian_window(:),1,size(vec_patches_Y,2)), nn, eps, false);
    close(anno);
    display(['time taken to calculate nearest neignbor at all pixels: ' num2str(cputime-t)]);
    clear vec_patches_LR;
    clear x y gaussian_window vec_patches_LR;
    
    %% updating weight_mat_example and I_m_example
    
    B_Y = dim(2)-2*boundary;
    B_LR=size(img_LR,2)-2*boundary;
            
    boundary_scale=fac*boundary;
    [x,y]=meshgrid(-boundary_scale:boundary_scale,-boundary_scale:boundary_scale);
    gaussian_window=exp(-(x+y).^2/(2*sigma_PSF^2*(scale-l)));

    for i=1:size(idx,2)%for each patch in I{l}
        A = i - 1;
        m_Y = 1 + boundary + floor(A/B_Y); n_Y = 1 + boundary + rem(A,B_Y);
        for j=1:nn %search for NN in lower resolution image
        
            A = double(idx(j,i)) - 1;
            m_LR = 1 + boundary + floor(A/B_LR); n_LR = 1 + boundary + rem(A,B_LR);
            
            m1=(m_LR-1)*alpha^(scale-l)+1;%coordinates of HR in I{l}
            n1=(n_LR-1)*alpha^(scale-l)+1;
            
            HR_patch_in_Y=Y(m1-boundary_scale:m1+boundary_scale,n1-boundary_scale:n1+boundary_scale);
            HR_patch_in_Y=HR_patch_in_Y-mean2(HR_patch_in_Y)+mean2(Y(m_Y-boundary:m_Y+boundary,n_Y-boundary:n_Y+boundary));
            
            m1=(m_Y-1)*alpha^(scale-l)+1;%coordinates of HR in I{scale}
            n1=(n_Y-1)*alpha^(scale-l)+1;
            dist=double(dst(j,i));
            
            
            weight_window=ones(size(HR_patch_in_Y)).*gaussian_window*exp(-dist/(sigma_sq(i,1)));%(1/(dist+1))%%*(scale-l)
            
            %place HR patch constraint 
            I_m_example{scale-l,1}(m1-boundary_scale:m1+boundary_scale,n1-boundary_scale:n1+boundary_scale)=...
                I_m_example{scale-l,1}(m1-boundary_scale:m1+boundary_scale,n1-boundary_scale:n1+boundary_scale)+HR_patch_in_Y.*weight_window;
            
            weight_mat_example{scale-l,1}(m1-boundary_scale:m1+boundary_scale,n1-boundary_scale:n1+boundary_scale)=...
                weight_mat_example{scale-l,1}(m1-boundary_scale:m1+boundary_scale,n1-boundary_scale:n1+boundary_scale)+weight_window;
            
            
        end
    end
    
    
            if (numel(find(weight_mat_example{scale-l,1}==0))~=0) || (numel(find(isnan(weight_mat_example{scale-l,1})))~=0)
                [x y]=meshgrid(1:size(Y,2),1:size(Y,1));
                [xi yi]=meshgrid(1:1/fac:size(Y,2),1:1/fac:size(Y,1));
                img=zeros(round(fac*z1),round(fac*z2));
                
                zzz=interp2(x,y,Y,xi,yi,'cubic');
                temp_img=double(imresize_old(Y,fac,'bicubic'));
                img(end-fac+1:end,:)=temp_img(end-fac+1:end,:);
                img(:,end-fac+1:end)=temp_img(:,end-fac+1:end);

                clear temp_img;
                img(1:size(zzz,1),1:size(zzz,2))=zzz;
                clear zzz;
                
                weight_mat_example{scale-l,1}((isnan(weight_mat_example{scale-l,1})))=0;
                I_m_example{scale-l,1}((weight_mat_example{scale-l,1}~=0))=...
                    I_m_example{scale-l,1}((weight_mat_example{scale-l,1}~=0))./weight_mat_example{scale-l,1}((weight_mat_example{scale-l,1}~=0));

                I_m_example{scale-l,1}((weight_mat_example{scale-l,1}==0))=img((weight_mat_example{scale-l,1}==0));

                
            end

end