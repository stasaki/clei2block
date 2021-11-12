
library(tidyverse)
data_type="mRNA"
res_pca <- readRDS(paste0("/mnt/new_disk/007_iPWAS/Analysis/01_model/Data/07_ml_data_v3/pca_",data_type,".rds"))


compute_PCs = function(data,res_pca){
  # data is a gene-by-sample matrix with gene nemes as row names
  # res_pca is a list contining rotation matrix, and a data.frame storing scaling factors for gene expression values  
  indx = match(res_pca[[2]]$gene_id,rownames(data))
  res_pca[[1]] = res_pca[[1]][indx[!is.na(indx)],]
  res_pca[[2]] = res_pca[[2]][indx[!is.na(indx)],]
  data = data[indx[!is.na(indx)],]
  
  data_tmp = sweep(data, 1, res_pca[[2]]$center,FUN = "-")    
  data_tmp = sweep(data_tmp, 1, res_pca[[2]]$scale,FUN = "/")    
  
  dr_vec = t(data_tmp)%*%res_pca[[1]]
  
  return(dr_vec)
}

# generate demo-data
data = matrix(rnorm(res_pca[[1]]%>%nrow()* 150),nrow=res_pca[[1]]%>%nrow(),ncol=150)
rownames(data) = res_pca[[1]]%>%rownames()
colnames(data) = paste0("sample",1:150)


PC_vectors = compute_PCs(data,res_pca = train_pca)%>%
  as.data.frame()%>%
  rename_all(.funs = list(~paste0(data_type,"_",.)))%>%
  mutate(IID=rownames(.))

head(PC_vectors)
