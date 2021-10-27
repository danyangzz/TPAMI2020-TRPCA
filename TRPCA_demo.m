% - Title: Truncated Robust Principle Component Analysis with A General Optimization Framework
% - Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020
% - Author: Feiping Nie, Danyang Wu, et.al.
% - Contact: If you have any questions about this work, please feel free to contact 
%           danyangwu41x@mail.nwpu.edu.cn, we will response to you as soon as possible.
% -If this code is helpful to you, please kindly cite our paper:
%  "Truncated Robust Principle Component Analysis with A General Optimization Framework"    
% -------------------------------------------------------------------------
clc;
clear;
close all;

%--[Load data]---

load('ATT_data_block_1d2.mat');   % X0: raw data X: occlusion data idx: occlusion data number 
[d,n] = size(X);

       
%---¡¾variable storage¡¿---
par_number = 5;

etopkRPCA = zeros(1,par_number);  % reconstruction error (RE)
etopkRPCA_oc = zeros(1,par_number); % RE for clean images
etopkRPCA_clean = zeros(1,par_number); % RE for occlusion images

k_topkRPCA_black = [9,9,9,7,9]; % truncated parameter 

%---¡¾run¡¿---
for i = 1 : par_number
    feature_num = 10*i;
        
    % RPCA-topk 
    j = k_topkRPCA_black(i);
    rate =  0.1 + 0.05*(j-1);
    k = ceil(rate*n);  
    [W, b, obj, dn] = TRPCA_fun(X, feature_num,k);
    Xr_Topk = W*W'*(X-b*ones(1,n)) + b*ones(1,n);
    a = X0 - Xr_Topk;
    etopkRPCA(i) = sum(sqrt(sum(a.*a,1)));
    a = a(:,idx);
    etopkRPCA_oc(i) = sum(sqrt(sum(a.*a,1)));
    etopkRPCA_clean(i) = etopkRPCA(i) - etopkRPCA_oc(i);
end
    
   
    


    
 





