options(stringsAsFactors = FALSE)
library(tidyverse)

# Setup pytorch environment ####
# docker image: pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime OR Google Cloud instance image: c2-deeplearning-pytorch-1-2-cu100-20191005
# GPU: nvidia-tesla-t4
# Boot-disk-size: 50 GB
# CPU: 8 cores"

# Please specify file location ####
clei2block_loc = "./functions/" # available at https://github.com/stasaki/clei2block/functions/
data_loc = "./Input_train_anonymous/" # available at https://www.synapse.org/#!Synapse:syn23667887
out_loc = "./ROSMAP_model-out/"
python_loc = "/opt/anaconda3/bin/python" # depends on your enviroment


# Train ####
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
  mutate(script=paste0(clei2block_loc,"/clei2block.py"))
Z_dim=100
h_dim=800

lapply(1:nrow(jobs),
       function(n_job){
         sub_out_loc=paste0(out_loc,paste0(jobs$lr_name[n_job],"_",jobs$vae_input[n_job]),"/")
         dir.create(sub_out_loc,showWarnings = F,recursive = T)
         if(file.exists(paste0(sub_out_loc,"/4/test_prediction.npy"))){
           return()
         }
         cmd = paste0(c(python_loc," ",jobs$script[n_job],data_loc,sub_out_loc,Z_dim,h_dim,
                        "none", jobs$vae_input[n_job], paste0(unlist(jobs$lr_input[n_job]),collapse = ",")),collapse = " ")
         write.table(cmd,file=paste0(sub_out_loc,"/cmd.sh"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)
         system(cmd)
       })


# Merge output ####
use_samples <- paste0(data_loc,"/test_y_col.txt.gz")%>%
  read.delim(.,header = F)%>%.$V1%>%
  as.character(.)
protein_id = paste0(data_loc,"/test_y_row.txt.gz")%>%
  read_delim(.,col_names =F,delim = "\t",
             col_types = cols(
               X1 = col_character(),
               X2 = col_character()
             ))%>%
  dplyr::rename(probe_id=X1,gene_id=X2)

ensemble_out = paste0(out_loc,"/ensemble/")
dir.create(ensemble_out,showWarnings = F,recursive = T)

library(reticulate)
np=import("numpy")

lapply(1:nrow(jobs), function(n_job){
    lapply(0:4, function(iter){
      sub_out_loc=paste0(out_loc,paste0(jobs$lr_name[n_job],"_",jobs$vae_input[n_job]),"/",iter,"/")
      data = np$load(paste0(sub_out_loc,"/test_prediction.npy"))%>%
        as.matrix()%>%t()
      colnames(data) = use_samples
      data = bind_cols(protein_id,as_tibble(data))
      data%>%
        gather(SAMPLE.ID,pred,-probe_id,-gene_id)%>%
        return()
    })%>%bind_rows()%>%
      group_by(probe_id,gene_id,SAMPLE.ID)%>%
      summarise(pred=mean(pred))%>%
      return()
})%>%bind_rows()%>%
  group_by(probe_id,gene_id,SAMPLE.ID)%>%
  summarise(pred=mean(pred)) -> data
saveRDS(object = data,file = paste0(ensemble_out,"predicted_proteome.rds"))


