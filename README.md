clei2block: deep-neural protein translation
========
![clei2block Logo](logo1.png)


What is clei2block?
=================

`clei2block` is a method to predict protein abundance across samples from RNA-seq data. Due to the complexity of gene regulation, the discrepancy between mRNA and protein levels has been a standing question since both genome-wide measurements became available. There are two types of mRNA-vs-protein relationships: (i) across genes within a sample (ii) across samples for each gene. Our method models the latter relationship using deep-neural networks.

Installation
============

### Requirements

  * A system equipped with Nvidia GPU
  * [Docker](https://www.docker.com/whatisdocker/)
  * [R](https://www.r-project.org) version 3.6.2 or later
  
### Download GitHub repository

    git clone https://github.com/stasaki/clei2block.git

Working on a demo data
============

R notebook for this tutorial is available at `code/demo-code.html` and the expected outcome is `code/demo-code-expected-output.nb.html`

### Training model
Change current working directory to clei2block

    cd clei2block 
    
Command line arguments for clei2block are as follows 

    python clei2block.py <path to input data> <path to output> <dimensionality of the latent space> <dimensionality of hidden layer> <if you want to apply the model to other data specify the path to the data (Optional)> <input features for latent embedding> <input features for linear regression>

Inputs

  * latent embedding
    * `train_vae_x_mRNA.txt.gz`: input feature matrix for latent embedding (gene x sample)
    * `train_vae_x_mRNA_col.txt.gz`: sample id
    * `train_vae_x_mRNA_row.txt.gz`: gene id

  * linear regression module
    * `train_lr_x_mRNA.txt.gz`: input feature matrix for linear regression (gene x sample)
    * `train_lr_x_mRNA_col.txt.gz`: sample id
    * `train_lr_x_mRNA_row.txt.gz`: gene id

  * protein outcome
    * `train_y.txt.gz`: actual protein matrix (gene x sample)
    * `train_y_col.txt.gz`: sample id
    * `train_y_row.txt.gz`: gene id

Sample order must be identical for all inputs. Gene order must be identical for linear regression inputs and protein outcome. Test data should follow the same format, but the file name begins with `test` instead of `training`. 

Run clei2blck model with the demo data through docker

    time docker run --rm --gpus all -v $(pwd)/../:/var  pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime  /bin/bash -c "pip install pandas; python /var/functions/clei2block.py /var/demo-data/ /var/demo-out/ 100 800 none mRNA mRNA"
    
The training the model takes about 3 minutes. This script run clei2block 5 times and the results will be in subdirectories from `0` to `4`. In each subdirectory, you will see the following outputs.

  * `trained_model.pt`: trained pytorch model
  * `train_loss.txt`: loss for training data
  * `valid_loss.txt`: loss for validation data
  * `val_index.txt`: sample index used for validation data
  * `test_prediction.npy`: predicted protein values for testing data stored as numpy array
  
### Check if the loss is decreased as expected
Launch R

    R
    
Install required R libraries
```R
install.packages("ggplot2")
install.packages("dplyr")
install.packages("tidyr")
install.packages("reticulate")
np=import("numpy")
```
Run the following code in R
```R
library(ggplot2)
library(dplyr)
library(tidyr)

# collect training loss
list.files("../demo-out/",pattern = "train_loss.txt",full.names = TRUE,recursive = TRUE)%>%
  lapply(., function(x){
    read.delim(x,sep = ",",header = F)%>%
      mutate(trial=dirname(x)%>%basename(),
             step=1:n())%>%
      dplyr::select(trial,step,train_loss = V1)%>%
      return()
  })%>%
  bind_rows() -> loss_ds

# collect validation loss
list.files("../demo-out/",pattern = "valid_loss.txt",full.names = TRUE,recursive = TRUE)%>%
  lapply(., function(x){
    read.delim(x,sep = ",",header = F)%>%
      mutate(trial=dirname(x)%>%basename(),
             step=1:n())%>%
      dplyr::select(trial,step,valid_loss = V1)%>%
      return()
  })%>%
  bind_rows()%>%
  inner_join(loss_ds,.,by=c("trial","step"))%>%
  group_by(trial)%>%
  filter(1:n() <= which(valid_loss==min(valid_loss)))-> loss_ds

loss_ds%>%gather(loss_type,loss,-trial,-step)%>%
    ggplot(.,aes(as.factor(step),loss,color=loss_type))+
  geom_boxplot()+
  xlab("Training epochs")
```

### Check the model performance for testing data
Run the following code in R

```R
library(reticulate)
np=import("numpy")

# collect test prediction
list.files("../demo-out/",pattern = "test_prediction.npy",full.names = TRUE,recursive = TRUE)%>%
  lapply(., function(x){
    np$load(x)%>%
      as.matrix()%>%
      t()%>%
      return()
  }) -> pred_protein

# consensus prediction
for(i in 2:length(pred_protein)){
  pred_protein[1][[1]]=pred_protein[1][[1]]+pred_protein[i][[1]]
}
pred_protein = pred_protein[1][[1]] # this is a protein by sample matrix predicted by the clei2block model


# read actual measurements
acutal_protein = read.delim("../demo-data/test_y.txt.gz",header = F)%>%as.matrix()
acutal_mRNA = read.delim("../demo-data/test_lr_x_mRNA.txt.gz",header = F)%>%as.matrix()

# calculate correlation between predicted protein and acutal protein measurement for each gene across samples
lapply(1:nrow(acutal_protein), function(i){
  cor(pred_protein[i,],acutal_protein[i,])
})%>%unlist() -> pre_cor

# calculate correlation between acutal protein measurement and thier corresponding mRNA for each gene across samples
lapply(1:nrow(acutal_protein), function(i){
  cor(acutal_mRNA[i,],acutal_protein[i,])
})%>%unlist() -> mRNA_cor

# visualize correlation
ds = data.frame(cor=pre_cor,
           data = "predicted protein")
ds = bind_rows(data.frame(cor=mRNA_cor,
           data = "actual mRNA"),ds)
ds%>%
  ggplot(.,aes(data,cor,fill=data))+
  geom_boxplot()+ylab("Correlation with actual protein")
```

How to use a pretrained model
============

You can use a pretrined model with the following command. 

    python clei2block-predict.py <path to input data> <path to output> <path to pretrained model> <input features for latent embedding> <input features for linear regression> 
    
The order of input variables must be identical with the input used to train the model. Also if you use multiple data files for the input of latent embedding or linear regression, the order of files in the command must be identical with the training.

To apply a pretrained model to the demo-data through docker, run the following command.

    time docker run --rm --gpus all -v $(pwd)/../:/var  pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime  /bin/bash -c "pip install pandas; python /var/functions/clei2block-predict.py /var/demo-data/test_ /var/demo-out-predict/ /var/demo-out/0/trained_model.pt mRNA mRNA"
    
The training the model takes about 3 minutes. You will see the following outputs.

  * `predicton.npy`: predicted protein values stored as numpy array
  
Download models trained with ROSMAP brain data
============
The clei2block models for ROSMAP brain data are available at http://dx.doi.org/10.7303/syn23624037. 

You can programmatically download data and models via Synapse API client

    pip install synapseclient

    mkdir ROSMAP_model
    cd ROSMAP_model
    synapse get -r syn23624087 
    cd ..
    
    mkdir Input_predict_anonymous
    cd Input_predict_anonymous
    synapse get -r syn23667843
    cd ..
    
    mkdir Input_train_anonymous
    cd Input_train_anonymous
    synapse get -r syn23667887
    cd ..

You can build or use ROSMAP clei2block model using `code/train.R` and `code/predict.R`.

Access to ROSMAP/MSBB brain data
============
Data used in these analyses, the predicted proteome data, and the estimated pseudotimes can be requested at the RADC Resource Sharing Hub at www.radc.rush.edu or downloaded from the Synapse repository (http://dx.doi.org/10.7303/syn3219045, http://dx.doi.org/10.7303/syn3159438). The RNA-seq and protein data are available via the AD Knowledge Portal (https://adknowledgeportal.org). The AD Knowledge Portal is a platform for accessing data, analyses, and tools generated by the Accelerating Medicines Partnership (AMP-AD) Target Discovery Program and other National Institute on Aging (NIA)-supported programs to enable open-science practices and accelerate translational learning. The data, analyses, and tools are shared early in the research cycle without a publication embargo on secondary use. Data is available for general research use according to the following requirements for data access and data attribution (https://adknowledgeportal.org/DataAccess/Instructions).
