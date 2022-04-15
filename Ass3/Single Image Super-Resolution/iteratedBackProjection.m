function [corrected_image]=iteratedBackProjection(I_m,I_l,scale_change,sigma_PSF,PSF_size,alpha)


%project the higher resolution image I{m,1} onto lower resolution image
%I{l} and correct for error iteratively.

corrected_image=I_m;%initialize the corrected image with input image

lambda = 0.1; % define the step size for the iterative gradient method
max_iter = 100;% max number of iterations
iter = 1;

corrected_image_prev = corrected_image;
E = [];
psf_size=PSF_size*scale_change;
PSF=fspecial('gaussian',psf_size,sigma_PSF*sqrt(scale_change));

while (iter<max_iter)
    iter;
    filtered_HR = imfilter(corrected_image, PSF, 'symmetric','same');
    temp=filtered_HR(1:alpha^scale_change:end,1:alpha^scale_change:end)-I_l;
    
    err=imfilter(upsample(upsample(temp,alpha^scale_change)',alpha^scale_change)',PSF','symmetric','same');        
    
    corrected_image = corrected_image - (lambda) * err;
    
    change = norm(corrected_image-corrected_image_prev)/norm(corrected_image);
    E=[E; iter change];
    if iter>3 
      if abs(E(iter-3,2)-change) <1e-3
         break  
      end
    end
    corrected_image_prev = corrected_image;
    iter = iter+1;
end
corrected_image=corrected_image*mean2(I_l)/mean2(corrected_image);