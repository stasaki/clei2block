options(stringsAsFactors = FALSE)
library(tidyverse)

# Setup pytorch environment ####
# docker image: pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime OR Google Cloud instance image: c2-deeplearning-pytorch-1-2-cu100-20191005
# GPU: nvidia-tesla-t4
# Boot-disk-size: 50 GB
# CPU: 8 cores"

# Please specify file location ####
clei2block_loc = "./functions/" # available at https://github.com/stasaki/clei2block/functions/
rosmap_model_loc = "./ROSMAP_model/" # available at https://www.synapse.org/#!Synapse:syn23624087
data_loc = "./Input_predict_anonymous/" # available at https://www.synapse.org/#!Synapse:syn23667843
out_loc = "./out/"
python_loc = "/opt/anaconda3/bin/python" # depends on your enviroment

# Predict ####
vae_input=c("mRNA","mRNAPCA",
            "premRNA","premRNAPCA",
            "utr","utrPCA")
lr_input = list()
lr_input[["all"]]=c("mRNA","premRNA","utr")
lr_input[["mRNA"]]=c("mRNA")

jobs = data.frame(vae_input)%>%
  as_tibble()%>%
  mutate(lr_input=list(lr_input))%>%
  unnest()%>%
  mutate(lr_name = names(lr_input))%>%
  mutate(script=paste0(clei2block_loc,"/clei2block-predict.py"))%>%
  mutate(fset=paste0(lr_name,"_",vae_input))

lapply(1:nrow(jobs), function(i){
  lapply(1:10, function(n_split){
    lapply(0:4, function(iter){
      model_loc = paste0(rosmap_model_loc,jobs$fset[i], "/trained_model_fold",n_split,"_iter",iter,".pt")
      sub_out_loc=paste0(out_loc,"/fold",n_split,"/",jobs$fset[i],"/",iter,"/")
      dir.create(sub_out_loc,showWarnings = F,recursive = T)
      
      if(file.exists(paste0(sub_out_loc,"/y_prediction.txt.gz"))){
        return()
      }
      cmd = paste0(c(python_loc," ",jobs$script[i],data_loc,sub_out_loc,model_loc,jobs$vae_input[i], unlist(jobs$lr_input[i])),collapse = " ")
      write.table(cmd,file=paste0(sub_out_loc,"/cmd.sh"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)
      system(cmd)
    })
  })
})


# Merge output ####
use_samples <- read.delim(paste0(data_loc,"/lr_x_mRNA_col.txt.gz"),header = F)$V1
ensemble_out = paste0(out_loc,"/ensemble/")
dir.create(ensemble_out,showWarnings = F,recursive = T)

protein_id = readRDS(paste0(data_loc,"protein_id.rds"))
lapply(1:nrow(jobs), function(i){
  lapply(1:10, function(n_split){
    lapply(0:4, function(iter){
      sub_out_loc=paste0(out_loc,"/fold",n_split,"/",jobs$fset[i],"/",iter,"/")
      data = data.table::fread(paste0(sub_out_loc,"/y_prediction.txt.gz"),
                               sep = "\t")%>%
        as.matrix()%>%t()
      colnames(data) = use_samples
      data = bind_cols(protein_id,as_tibble(data))
      data%>%
        gather(SAMPLE.ID,pred,-V1,-V2)%>%
        return()
    })%>%bind_rows()%>%
      group_by(V1,V2,SAMPLE.ID)%>%
      summarise(pred=mean(pred))%>%
      return()
  })%>%bind_rows()%>%
    group_by(V1,V2,SAMPLE.ID)%>%
    summarise(pred=mean(pred))%>%
    return()
})%>%bind_rows()%>%
  group_by(V1,V2,SAMPLE.ID)%>%
  summarise(pred=mean(pred)) -> data
saveRDS(object = data,file = paste0(ensemble_out,"predicted_proteome.rds"))

