# ViT-Visual-Interpretability

This is the official implementation of the paper: Multiplex Network-Based Representation of Vision Transformers for Visual Explainability.

## Abstract



![Multilayer creation](./readme_imgs/Workflow_1.pdf)

![Heatmap generation](./readme_imgs/Workflow_2.pdf)




## Usage

This repository can be directly downloaded and executed locally. The required libraries are displayed in Section [Requirements](#requirements)

In the **Implementation** folder we provide a file called **Usage_example.ipynb** where the user can run the cells and visualize insertion AUC, deletion AUC and heatmaps of one or more images. 
The code can be used both for ViT and DeiT explaination. In addition, this folder contains the following python files:
- **utils**: contains some functions used for the visualization of AUCs and heatmaps.
- **feature_extractor.py**: contains the definition of the feature extractor class used by ViT and DeiT models.
- **hook.py**: contains the definition of the hook classes used by ViT and DeiT models.
- **mutex.py**: contains the implementation of the model described in the paper.

The **imgs_idx** file contains the indexes of the images used for testing our approach, which are selected from the ImageNet 2012 validation dataset. To download the dataset you need to login on [ImageNet site](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and click on ILSVRC2012, after this download the 'Development kit (Task 1 & 2)', that contains the ground truth labels, and the 'Validation images (all tasks)'.


## Parameters 
In the **Usage_sample** file the user can modify the following parameters:

### Initialization parameters
**`model`**: represents the model of which we want to make explainability; in this demo this value can be _'vit'_ or _'deit'_;  
**`device`**: device in which the model will be used; this can be _'cpu'_ or _'cuda:0'_;  

### Call Parameters
**`min_cut`**: for each layer $L$, we delete the arcs with an attention value less than $min\\_cut*max(L)$, where $max(L)$ is the maximum value of attention for the layer $L$;  
**`token_ratio`**: percentage of patches which are set to 0 during the binary masks creation;  
**`starting_layer`**: layer from which are calculated the metrics used to construct the masks;  
**`img_path`**: path of the image we want to explain;  
**`label`**: label on which we want to do explaination; this value is optional and if is not provided the model will give the heatmap associated with the predicted class;  




## Requirements <a name="requirements"></a>

In our notebook we used the following libraries:
```
PIL=9.2.0  
scipy=1.9.3  
transformers=4.30.2  
torch=2.1.2+cu121  
torchvision=0.16.2+cu121  
sklearn=1.4.1.post1  
numpy=1.24.4  
pandas=2.1.4  
matplotlib=3.8.2
seaborn=0.13.1  
```

## Citation

If you use this model for your research please cite our paper.
