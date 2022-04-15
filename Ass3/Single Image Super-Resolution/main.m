%%% Input image=im
%%% resolution increase factor =alpha
%%% M governes maximum resolution increase =alpha^M
%%% If you want to increase the resolution by 2x, provide alpha=2, M=1
%%% The results largely depend on the variance of blur kernel, size of blur
%%% kernel, although we have found these parameters when used in default
%%% (as they are) to give goo results.

%%% The most important factor for governing quality of results is sigma_sq
%%% for each query patch. You may use adaptive sigma as a function of patch
%%% variance etc. However, keeping sigma_sq=constant(between 10 to 150) 
%%% provides quiet good results.
%%% Insight into sigma_sq values: For the present implementation, sigma_sq
%%% decides on the relative effects of classical and example based
%%% resolution constraints. Higher sigma_sq value may give you more effect 
%%% to example based technique which may smooth out some details.
%%% If you have efficeint subpixel alignment tool and enough similar %
%%% patches are found, then use that for subpixel alignment and you can try 
%%% adjusting sigma_sq to have more effect of classical SR constraints.
%%% In summary, try varying sigma_sq value according to input scale in   
%%% range 10 to 150. You will observe which value of sigma_sq works the 
%%% best. Also, getting good results may depend on the skill of the user 
%%% and there may be some other parameters and experimentaion required to 
%%% get desired results. 

close all;
clear all;
im =imread('inp1_forest.png');

total_time_start=cputime;

if (length(size(im))==3)
    im = rgb2ycbcr(im);
    Y = double(im(:,:,1));
else
    Y = double(im);
end

dim = size(Y);

global PSF_size;
global length_patch;
length_patch= 5;%5x5 patch, must be odd number
global size_patch;
size_patch = length_patch*length_patch;
global boundary;
boundary = floor(length_patch/2);

alpha=2; %scale change factor

M=2; %Final resolution increase = alpha^M

sigma_PSF=sqrt(.3*alpha); %sigma of PSF for a scale change of alpha^1

I=cell(2*M+1,1);%cell structure to store the images I(-M),I(-M+1)...,I(0),...I(M)
%I(0)=original input image, I(m)=required HR image

I{M+1,1}=Y; %I(0)
PSF_size=[2*alpha+1 2*alpha+1];

%storing downsampled version of I*Blur(sl) in I(1) to I(M)
for i=-1:-1:-M
    PSF=fspecial('gaussian',PSF_size*abs(i),sigma_PSF*sqrt(abs(i)));
    temp=imfilter(Y,PSF,'symmetric','same');
    I{M+1+i,1}=temp(1:alpha^abs(i):end,1:alpha^abs(i):end);
end

%% construct increasing resolution images in steps
for m=M+2:2*M+1 %constructing I{m} %I(0) is stored in I{M+1}
    
    %Constructing I{m} now
    
    display(['solving for I(' num2str(m-M-1) ')'])
    I_m_classical=cell(m-1-M,1);
    weight_mat_classical=cell(m-1-M,1);
    weight_mat_example=cell(m-1-M,1);
    I_m_example=cell(m-1-M,1);
    
    %apply constraints
    for l=M+1:m-1 %in order to construct I{m} we use all the HR images 
        %recovered so far from I{M+1} to I{m-1} 
        
        [vec_patches_l]=createPatchVector(I{l},boundary);
    
        sigma_sq=zeros(size(vec_patches_l,2),1);
        for patches = 1:length(sigma_sq)
            sigma_sq(patches)=19; %%The most important tuning parameter
            %sigma_sq(patches)=15;
            %sigma_sq(patches)=var(vec_patches_l(patches))+1;
            
        end
        clear vec_patches_l;
        
        
        %% Classical SR constraints
        display(['classical SR' num2str(l-M)])
        [weight_mat_classical{l-M,1} I_m_classical{l-M,1}]=classicalSR((m-l)*sigma_PSF,I{l},alpha^(m-l),boundary,m-l,sigma_sq); 
        
        %% Example based SR constraints
        display(['example SR' num2str(l-M)])
        
        [weight_mat_example{l-M,1} I_m_example{l-M,1}]=exampleSR(sigma_PSF,I,boundary,l,m,alpha,sigma_sq);
    end
    
    %Solve the constraints induced by classical and example based SR using
    %% iterative least square
    display(['Solving LS for I(' num2str(m-M-1) ')'])
       
    [I{m,1}]=LS_solve(I,M,m,sigma_PSF,weight_mat_classical,I_m_classical,weight_mat_example,I_m_example,alpha);
    
    
    %% Iterated Backprojection
    [I{m,1}]=iteratedBackProjection(I{m,1},I{l},(m-l),sigma_PSF,PSF_size,alpha);
    
end
    
%% display output
im_bicubic=imresize(im,alpha^M,'bicubic');
HR=im_bicubic;
HR(:,:,1)=I{2*M+1,1};
if size(HR,3)==3
HR=ycbcr2rgb(HR);
end
figure,imshow(HR)
title('Super resolved Image')
if size(HR,3)==3
im_bicubic=ycbcr2rgb(im_bicubic);
end
figure,imshow(im_bicubic)
title('Bicubic interpolated version')

display(['Total time taken by the algorithm' num2str(cputime-total_time_start)])
 