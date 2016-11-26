%% nnw 2015.11 Detection of varied defects in diverse fabric images via modified RPCA with noise term and defect prior
setup
path='../data/temp_7_27';% input data
write_patch = '..\all_results\';%output data
files = dir(path);
k=1;
for  file_count = 3: length(files)
    image_name=[path '\' files(file_count).name];
    %image_name=[files(file_count).name];
    I = imread(image_name);
    dot_id = strfind(image_name , '.') ; sprit_id = strfind(image_name , '\') ;
    name = image_name(sprit_id(end)+1: dot_id-1);
    [m , n , cha] = size(I) ;
    if cha > 1
        I = rgb2gray(I) ;
    end
    I = double(I) ;
    I = I/255;
    %% 尝试加入余弦基效果并不好
    % B1=dctmtx(size(I,1));
    % B2=dctmtx(size(I,2));
    %% rpca
    alfa=0.03;
    bita=0.2;
    [A E] = inexact_alm_rpca(I , alfa) ;
    write_name = [write_patch name];
    imwrite(mat2gray(abs(E)),[write_name 'rpca.png']);
    %% N-RPCA
    [A E F] = inexact_alm_NRPCA(I , alfa,bita) ;
    write_name = [write_patch name];
    imwrite(mat2gray(abs(E)),[write_name 'nrpca.png']);
    %% PN-RPCA
    patch_size=16;
    over_size=8;
    %提取texton特征作为先验信息
    [patch_id] = compute_uniformly_size_patch(I , patch_size , over_size) ; %bar 下采样160 子块16,8 start 下采样160块 子块12 6
    patch_prior3 = patch_blemish_prior_textons( I , 'rand' , patch_id) ;% perior = recover_saliency( I, patch_prior3 , patch_id);
    perior_norm=mat2gray(patch_prior3);
    T=graythresh(perior_norm);
    P=im2bw(perior_norm,T);
    prior = recover_saliency( I , P , patch_id);
    [A E F]= inexact_alm_PNRPCA(I,alfa,bita,prior) ;
    imwrite(mat2gray(abs(E)),[write_name 'pnrpca.png']);
%% RRSVD   
        mysvd(image_name);  
%% wavelet %选取的水平方向
        mydwt2(image_name)   %包括手动阈值和自动阈值
%% LBP    
        Wd =  26;
        r1 = 1; r2 = 2;
        wm1 = 8 ;wm2 = 16 ;
        way = 'riu2';
        toversize = 2; oversize = 13;
        thre_mul = [0.8 1.0 1.2 1.4 1.5 1.7 2] ;
        modifymain(image_name,Wd,r1,r2,wm1,wm2,way,toversize,oversize , thre_mul);
end


