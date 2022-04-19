function [HR_final]=LS_solve(I,M,m,sigma_PSF,weight_mat_classical,I_m_classical,weight_mat_example,I_m_example,alpha)
%script for iterative least square solution
up_factor=alpha;

I_temp=I{m-1};
global PSF_size;
%% construct the bicubic interpolated version for intialization
[z1 z2]=size(I_temp);
[x y]=meshgrid(1:size(I_temp,2),1:size(I_temp,1));
[xi yi]=meshgrid(1:1/up_factor:size(I_temp,2),1:1/up_factor:size(I_temp,1));
img_bicubic=zeros(round(up_factor*z1),round(up_factor*z2));


zzz=interp2(x,y,I_temp,xi,yi,'cubic');
temp_img=double(imresize(I_temp,up_factor,'bicubic'));
img_bicubic(end-up_factor+1:end,:)=temp_img(end-up_factor+1:end,:);
img_bicubic(:,end-up_factor+1:end)=temp_img(:,end-up_factor+1:end);

clear temp_img;
img_bicubic(1:size(zzz,1),1:size(zzz,2))=zzz;
clear zzz;

%%  
HR_final=img_bicubic;
%Initialize HR with approximate soln (can also initialize HR with an
%approximate soln.) uncomment the two line below
%HR_final=Approx_soln(I,M,m,sigma_PSF,weight_mat_classical,I_m_classical,weight_mat_example,I_m_example,alpha);
%HR_final(HR_final==0)=img_bicubic(HR_final==0);

gamma=1;    
lambda = 0.1; % define the step size for the iterative gradient method
max_iter = 150;% max number of iterations
iter = 1;
       
Weight_mat=zeros(size(HR_final));

%convolving weight matrices with gaussian PSF with appropriate variance determined by scale gap 
for l=M+1:m-1
    %PSF=fspecial('gaussian',[5 5],var_PSF*alpha^(m-l));
    PSF=fspecial('gaussian',PSF_size*(m-l),sigma_PSF*sqrt((m-l)));        
    Weight_mat=Weight_mat+imfilter(weight_mat_classical{l-M,1},PSF','symmetric','same');
    
    for scale=l+1:m
        if (m-scale)==0
            PSF=1;
        else
            PSF=fspecial('gaussian',PSF_size*(m-scale),sigma_PSF*sqrt(m-scale));
        end
        tempp=imfilter(upsample(upsample(weight_mat_example{l-M,1}{scale-l,1},alpha^(m-scale))',alpha^(m-scale))',PSF','symmetric','same');
        Weight_mat=Weight_mat+tempp;

    end
end

while (iter<max_iter)
    iter;
    G = zeros(size(HR_final));%gradient 

    for l=M+1:m-1
        %error due to classical SR constraints
        PSF=fspecial('gaussian',PSF_size*(m-l),sigma_PSF*sqrt(m-l));        
        filtered_HR = imfilter(HR_final, PSF, 'symmetric','same');
        temp=weight_mat_classical{l-M,1}.*(filtered_HR-I_m_classical{l-M,1});%./(Num_patches+10^-3);
        temp=imfilter(temp,PSF','symmetric','same');
            
        G=G+temp;
    
        for scale=l+1:m
            %error due to example-based SR constraints
            fac=alpha^(m-scale);
            if (m-scale)==0
                PSF=1;
            else
                PSF=fspecial('gaussian',PSF_size*(m-scale),sigma_PSF*sqrt(m-scale));
            end
        
            filtered_HR = imfilter(HR_final, PSF, 'symmetric','same');
            temp=weight_mat_example{l-M,1}{scale-l,1}.*(filtered_HR(1:fac:end,1:fac:end)-I_m_example{l-M,1}{scale-l,1});
            temp=imfilter(upsample(upsample(temp,alpha^(m-scale))',alpha^(m-scale))',PSF','symmetric','same');
            G=G+temp;
        
        end
    end
    
    G((Weight_mat~=0))=G((Weight_mat~=0))./Weight_mat((Weight_mat~=0));
    
    G=gamma*G+(1-gamma)*HR_final;
    G_prev=G;
    HR_final = HR_final - (lambda) * G;
    
     iter = iter+1;
end

HR_final=HR_final*mean2(img_bicubic)/mean2(HR_final);