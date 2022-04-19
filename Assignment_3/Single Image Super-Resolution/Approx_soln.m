function [HR_final]=Approx_soln(I,M,m,sigma_PSF,weight_mat_classical,I_m_classical,weight_mat_example,I_m_example,alpha)
up_factor=alpha^(m-(M+1));
I0=I{M+1,1}; %Image at level0, original LR image

global PSF_size;
[z1 z2]=size(I0);
HR_final=zeros(round(up_factor*z1),round(up_factor*z2));
Weight_mat=zeros(size(HR_final));

%convolving weight matrices with gaussian PSF with appropriate variance determined by scale gap 
for l=M+1:m-1
    PSF=fspecial('gaussian',PSF_size*(m-l),sigma_PSF*sqrt((m-l)));        
    Weight_mat=Weight_mat+weight_mat_classical{l-M,1};
    HR_final=HR_final+weight_mat_classical{l-M,1}.*I_m_classical{l-M,1};
    
    for scale=l+1:m
        if (m-scale)==0
            PSF=1;
        else
            PSF=fspecial('gaussian',PSF_size*(m-scale),sigma_PSF*sqrt(m-scale));
        end
        tempp=imfilter(upsample(upsample(weight_mat_example{l-M,1}{scale-l,1},alpha^(m-scale))',alpha^(m-scale))',PSF','symmetric','same');
        Weight_mat=Weight_mat+tempp;
        temp=weight_mat_example{l-M,1}{scale-l,1}.*I_m_example{l-M,1}{scale-l,1};
        temp=imfilter(upsample(upsample(temp,alpha^(m-scale))',alpha^(m-scale))',PSF','symmetric','same');
        
        HR_final=HR_final+temp;
    end
end
HR_final((Weight_mat~=0))=HR_final((Weight_mat~=0))./Weight_mat((Weight_mat~=0));    
HR_final=HR_final*mean2(I0)/mean2(HR_final);