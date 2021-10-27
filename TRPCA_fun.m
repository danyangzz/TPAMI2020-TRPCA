% - Title: Truncated Robust Principle Component Analysis with A General Optimization Framework
% - Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020
% - Author: Feiping Nie, Danyang Wu, et.al.
% - Contact: If you have any questions about this work, please feel free to contact 
%           danyangwu41x@mail.nwpu.edu.cn, we will response to you as soon as possible.
% -If this code is helpful to you, please kindly cite our paper:
%  "Truncated Robust Principle Component Analysis with A General Optimization Framework"    
% -------------------------------------------------------------------------
function [W, b, obj, d, W_ini] = TRPCA_fun(X, feature_num, k, W0)
%--【Input】--
% X: dim*num data matrix, each column is a data point
% feature_num: the number of reduced dimensionality
% k: parameter of truncated leraning
% W0: Intialized Projection Matrix (In this code, we initialize d)
%--【Output】--
% W: learnt projection matrix
% b: learnt bias
% obj: objective value
% d: learnt smooth weights
% W_ini: Intialized projection matrix (In this code, we do not use it)

[dim n] = size(X);
%X = X - mean(X,2)*ones(1,n);

if nargin <= 3
    W = orth(rand(dim,feature_num));
    W_ini = W;
else
    W = W0;
end

d = ones(n,1);
p = ones(n,1);

for iter = 1:20  
        
    % for b 
    D = spdiags(sqrt(d),0,n,n);  
    b = X*d/sum(d);  
    
    % for W 
    A = X - b*ones(1,n);
    M = A*D;
   [W S T] = svds(M,feature_num);
    
    B = A - W*(W'*A);
    Bi = sqrt(sum(B.*B,1) + abs(eps))'; 
                                           
    % for \alpha
    p = zeros(n,1);
    [~,idx] = sort(Bi,'ascend'); 
    idx1 = idx(1:k,:);           
    p(idx1) = 1;                                                                                                                                                                                                                       % 将前K个的权置为1

    % for d  
    d = zeros(n,1);
    d(idx1) = 0.5./(Bi(idx1));
    
    obj(iter) = sum(p.*Bi);
    plot(obj);
end