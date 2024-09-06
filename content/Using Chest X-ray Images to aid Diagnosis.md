---
title: Using Chest X-ray Images to aid Diagnosis
draft: false
tags:
  - Projects
---

# Using Chest X-ray Images to aid Diagnosis 

# Introduction

Medical imaging diagnosis aids have the potential to greatly assist radiologists and improve patient outcomes. In particular, chest X-rays are critical for diagnosis of many common diseases, but they can be difficult to accurately interpret due to low contrast and visual subtlety of abnormalities. Computer vision techniques offer solutions by identifying visual indicators that may be difficult for the human eye to discern. Our project aims to create an assistive computer vision model that can highlight regions of interest in chest X-rays to improve radiologist workflow and diagnosis accuracy. This is an important capability that could reduce interpretation workload for radiologists while catching potential issues they may otherwise miss.

![400](https://production-media.paperswithcode.com/datasets/ChestX-ray14-0000001144-46559e6f_9iVbS0m.jpg)


# Related Work: 

### Using Convolutional Neural Networks (CNNs)
A number of studies have utilized Convolutional Neural Networks (CNNs) for the detection of COVID-19 from chest X-ray images. In [1], the authors propose a deep convolutional neural network trained on five open access datasets with binary output: Normal and Covid. The performance of the model is compared with four pre-trained CNN-based models (COVID-Net, ResNet18, ResNet, and MobileNet-V2). Similarly, [2] developed a new diagnosis platform using a deep CNN to assist radiologists with diagnosis by distinguishing COVID-19 pneumonia from non-COVID-19 pneumonia in patients at Middlemore Hospital based on chest X-rays classification and analysis. In [4], the researchers aim to train deep learning model that diagnoses with COVID-19. 

### Comparative Studies on Deep Learning Models for Chest X-ray Analysis
Comparative studies on the existing prominent methods have also been conducted. In [3], the comparison is done between binary classification performance of eight promiment deep learning models and compared across multiple metrics on the combined Pulmonary Chest Xrays dataset. 

### Using Transformer Architecture
In [5], the study conducted introduces a neural network that detects 14 pulmonary diseases using two branches: global (InceptionNetV3) and local (Inception modules and a modified Vision Transformer). The model achieved an AUC of 0.8012 and an accuracy of 0.7429, outperforming well-known classification models. \
Similar to this, another approach using transformer architecture is outlined in [6]. This paper proposed a new method, Multi-task Vision Transformer (MVC), for simultaneously classifying chest X-ray images and identifying affected regions. The method was built upon the Vision Transformer but extended its learning capability in a multi-task setting. The results showed the superiority of the proposed method over baselines on both image classification and affected region identification tasks.

### Our Contribution
Our original goal was to train a model like the models above and hopefully achieve better results. Then, we would attempt to use explainability methods to understand how the model classified each image. We hoped this procedure would produce novel results and lead to new contributions to the field. However, as will become evident in the rest of the report, we struggled to implement models that accomplished the tasks of previous works given our limited resources and expertise in this field. To produce novel results, we would need many more training resources to dedicate to hyperparameter optimization to reduce model overfitting.

# Method/Approach: 

### Dataset info

Below are a few examples of what the images in the dataset look like.

![](https://i.imgur.com/5S4MZ1J.png)

Our data, acquired from [7], was pre-split into train and test sets. This was to ensure that both subsets contained a similar distribution of dataset labels. The train and test set have 86524 and 25596 samples respectively. Each image has a resolution of 1024x1024 and is labeled into 1 of 15 categories. However, we scaled the image down to a resolution of 224x224 before inputting it into the model. The distribution of the training and test sets are seen below.

![](https://i.imgur.com/8PTnkXM.png)

![](https://i.imgur.com/uKNWymF.png)

As you can see, some labels hardly appear in the dataset. To make sure both distributions were similar, the dataset was pre-split into separate categories. 

### Switching to a Smaller Dataset

However due to this dataset being too large and training time taking too long, we switched to a smaller dataset [8]. The dataset is structured into three directories: ‘train’, ‘test’, and ‘val’, each containing subdirectories for two types of images: ‘Pneumonia’ and ‘Normal’. It comprises a total of 5,863 JPEG X-Ray images.

Selected from a retrospective cohort, the chest X-ray images (anterior-posterior view) belong to pediatric patients aged one to five years from the Guangzhou Women and Children’s Medical Center in Guangzhou. 

![](https://i.imgur.com/3FEWC09.png)

### Pre-processing

#### Smaller Dataset

For the smaller dataset, we resized the images to (224, 224) and converted them to RGB format. We also normalized the pixel values by dividing them by 255.0.
For the smaller dataset, we found 
#### Larger Dataset
As is typical with image classification tasks, we adding a few different augmentaiton to ouot dataset before training a model. We adding random flip, rotation, an affine trasnform, and normalized the values of each pixel between the range of [-1, 1]. For our best approach, we normalized each channel based on the mean and standard deviation of image net. 

### Model Architecture

#### Regular Dataset

We used three different models for the smaller dataset:

1. Custom CNN:
    - Our custom CNN consists of three convolutional layers with 32, 64, and 64 filters, respectively. Each convolutional layer is followed by a max pooling layer.
    - The output from the convolutional layers is flattened and passed through two dense layers with 64 and 2 units, respectively.
    - We used ReLU activation for the convolutional and dense layers, and softmax activation for the output layer.
2. Transfer Learning:
    - We used the ResNet152V2 model pre-trained on ImageNet as the base model.
    - We froze the base model layers and added custom layers on top for classification.
    - The custom layers include a global average pooling layer, a dense layer with 128 units and ReLU activation, a dropout layer with a rate of 0.1, and a final dense layer with 2 units and softmax activation.
3. Fine-tuning with Transfer Learning:
    - We used the same ResNet152V2 base model as in the transfer learning approach.
    - We unfroze the last 20 layers of the base model for fine-tuning.
    - The custom layers added on top are similar to the transfer learning approach.

![250](https://i.imgur.com/wLJkcQ0.png) ![250](https://i.imgur.com/MiAlLuk.png) 

*Model Architecture for Custom CNN* | *Model Architecture For Transfer Learning From Resnet*

#### Larger Dataset

For the larger dataset we tried to use densenet-121 pre-trained with image net weights. This DenseNet attemtps to classify each image into 1 of the specific 15 possible classes, as detalied above. A densenet is similar to a res-net however in addition to feeding the result forward arcross 1 layer, the densenet forwards to result to ALL downstream layers in a single dense block. Each element ina dense block contains convolutional layers of varying kernel sizes and strides. The DenseNet itself contains 4 dense blocks of varying sizes with normalization layers in between. Below is an image of a single dense block. 

![](https://i.imgur.com/n783jfn.png)

We expected this method to achieve similar results to various papers discussed in the related works section. However, we drastically underestimated the training resources needed for training and hyperparameter tuning. Overall, we spent over 20 hours of model training using A100 GPUs on Google Colab and could not achieve satisfactory results.  

We also attempted to train a resnet-50 and a custom ResNet with parameters chosen through hyperparameter optimization. 

### Intuition
The intuition behind why a DenseNet works better than other architectures is because of the nature of how the feed-forward layers work. In this way, each convolutional layer gets input from all the previous combinations of layers. Since convolutional layers can be described as feature extractors for images, each layer in a DenseNet receives a combination of all the features extracted from all previous layers. From that information, it can find more detailed patterns in the data. In basic ResNet, we only feed-forward the input across 1 layer. Therefore, a layer can only learn trends from the features extracted from the 1 previous layer instead of all previous layers. This is why DenseNets generally produce higher accuracy than other models. In addition, when explainability methods were used on a DenseNet, it would have produced potentially novel insights into how the model classified each X-ray. However, in our experiments, we were not able to successfully train a DenseNet or any other complex model, so we were unable to use these explainability techniques.


# Experiments

## Regular Dataset
We conducted experiments using three different models: custom CNN, transfer learning, and fine-tuning with transfer learning.

The input to the models is the smaller dataset with chest X-ray images of size (224, 224, 3) and corresponding labels (0 for normal, 1 for pneumonia).

The desired output is the classification of the images into two classes: normal and pneumonia.

We used validation accuracy and loss as the metrics for primary evaluation. To determine the best performing model, our metric was a combination of looking at the loss graphs, test and validation accuracy and the AUC score. 

### Custom CNN 
Our best approach and hyperparameters with our custom CNN yielded the following results.
![](https://i.imgur.com/x7hjtvC.png)

As we can see in the graph above. The loss function for training doesn't correspond well 
	

## Large Dataset
The models we attempted to train are a CNN, a custom ResNet, a ResNet-50, and a DenseNet-121. The two latter models used pre-trained weights from Image Net. In training the custom dataset, we used an automated hyperparameter tuning library called Optuna to tune the hyperparameters of our model before initial training. We also based other model training variables off previous works. We used cross entropy loss along with various types of optimization function, as well as methods that reduced the learning rate over time to prevent overfitting. 



# Results
  
### Regular Dataset

Our best performing model was fine-tuning with transfer learning. Below are some key graphs and metrics associated with evaluating the model. The evaluations are done on test split, which is separate from train and test.

![](https://i.imgur.com/KBPUbeB.png)

As we can see from the loss graph above, due to the finetuning at the end, our model tends to overfit on our training data. However, I got the highest validation accuracy with this approach.

![500](https://i.imgur.com/7Gi0tiG.png )

![400](https://i.imgur.com/Jm1rkM9.png)

The high AUC value = 0.99 on the ROC curve indicates a good classifier.


### Explainability Analysis

To gain insights into how the fine-tuned model makes predictions, we performed an explainability analysis using LIME (Local Interpretable Model-Agnostic Explanations) [9]. LIME is a technique that helps explain the predictions of black-box models by generating local explanations for individual instances.

We randomly selected a test image and generated an explanation for the model's prediction using LIME. The explanation highlights the regions of the image that contributed most to the model's decision.

The LIME explanation shows the true label, predicted label, and the probability assigned by the model to the predicted class. The image is overlaid with the segmentation mask generated by LIME, indicating the important regions.

![250](https://i.imgur.com/iplW0ub.png)


To further understand the model's decision-making process, we visualized the top 5 connected features that contributed to the predicted class.


![300](https://i.imgur.com/2IyjyjJ.png) ![300](https://i.imgur.com/MAn36k4.png)


The visualization of the top 5 connected features provides insights into the specific areas of the image that the model considers most relevant for the predicted class.
By using LIME, we can interpret the model's predictions and identify the regions of the chest X-ray images that are most informative for detecting pneumonia. 



###  Larger dataset


The baseline performance for multi-class classification on the NIH dataset is 84% accuracy on the testing set. This was obtained using a DenseNet-121 with pre-trained ImageNet weights.
Our performance:


| Model                   | Test Set Accuracy |     |
| :---------------------- | :---------------- | --- |
| Custom ResNet           | 37.4%             |     |
| Pretrained ResNet-50    | 38.7%             |     |
| Pretrained DenseNet-121 | 39.42%            |     |
| SOTA                    | 84.0%             |     |




Each of our model’s test accuracy is after 3 epochs.

Below is the graph of the accuracy of our DenseNet-121 model on the validation set per-label. As you can see, it is not just predicting one label and is learning, however, it still did not perform well. For a sanity check to make sure the model was training, we found the accuracy on the training set, which was 62.9%. This most likely means our model was overfitting the data. 

![](https://i.imgur.com/eoqzoeW.png)


To improve our results and reduce overfitting, we changed many hyperparameters of the model and used other techniques. We experimented with changing the learning rate, weight decay, momentum, and batch size parameters. We also experimented with different optimizers, such as RMSProp, SGD, Adam, and Adamax. In our custom ResNet, we changed the number of layers in each ResNet block. Finally, with our DenseNet-121 pre-trained model, we used a learning rate scheduler that dynamically changed the learning rate across different epochs. This, in theory, would help reduce overfitting when doing transfer learning. Within our dataset, we experimented with different image augmentations as well.

Our approach did not work and did not get close to the SOTA accuracy on the test set. Overall, we are not sure what went wrong, but we can tell that the model was overfitting. Our theory is that we did not train the models for long enough. However, this was hard to gauge because training 1 epoch took, on average, 70 minutes for each model, with more complex models taking longer. We tried to reduce the training set size and train the models for longer epochs, but that produced similar results. One other potential issue is that scaling the image down to 224x224 removed some information. However, the SOTA method did the same thing, so we don’t think that is the issue. We also tried other methods to reduce overfitting, like adding dropout layers. That still did not fix the issue. 


# Discussion
When using the NIH dataset, we were not able to create a model with high testing accuracy. While our model was not predicting the same label for every sample, and therefore learning nothing, it was getting a very low accuracy on the testing set across all epochs. We ended up not learning much from this dataset.

# Challenges Encountered 

One of the biggest challenges was simply the size of our dataset. Even after downloading it and uploading it to Google Drive, it took 20 minutes for Colab to load it into memory. We did mitigate this problem by streaming the data instead, which drastically reduced that time. 

Other challenges included the seemingly overfitting issue with all our models. Even when trying models of varying complexity and using techniques to reduce overfitting, nothing seemed to solve the problem. This may stem from the fact that the model took a while to train, sometimes upwards of 80 minutes per epoch.

If we were to start again, we would consider using code from a paper directly to verify we can successfully train a model on the data. Only after that would we consider moving to a more custom model. I would also start with smaller datasets, as we later discovered would be easier to use.  


# References

[1]A. Saxena and S. P. Singh, A Deep Learning Approach for the Detection of COVID-19 from Chest X-Ray Images using Convolutional Neural Networks. 2022.  \
[2]T. Gao and G. Wang, “Chest X-ray image analysis and classification for COVID-19 pneumonia detection using Deep CNN,” medRxiv, 2020, [Online]. Available: https://api.semanticscholar.org/CorpusID:221298503  \
[3] S. A. Shuvo, M. A. Islam, Md. M. Hoque, and R. B. Sulaiman, Comparative study of Deep Learning Models for Binary Classification on Combined Pulmonary Chest X-ray Dataset. 2023. \
[4] R. R. B. Negreiros, I. H. S. Silva, A. L. F. Alves, D. C. G. Valadares, A. Perkusich, and C. de Souza Baptista, “COVID-19 Diagnosis Through Deep Learning Techniques and Chest X-Ray Images,” SN Computer Science, vol. 4, pp. 1–12, 2023, [Online]. Available: https://api.semanticscholar.org/CorpusID:26080898 \
[5] A. Mezina and R. Burget, “Chest X-ray Image Analysis using Convolutional Vision Transformer,” Proceedings II of the 29st Conference STUDENT EEICT 2023: Selected papers., 2023, [Online]. Available: https://api.semanticscholar.org/CorpusID:260031717  \
[6] H. Tran, D. T. Nguyen, and J. Yearwood, MVC: A Multi-Task Vision Transformer Network for COVID-19 Diagnosis from Chest X-ray Images. 2023. \
[7] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE, 2017, https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset
[8] Mooney Paul, “Chest X-Ray images (Pneumonia).” [Online]. Available: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
[9] “Pneumonia Detection using CNN(92.6% Accuracy),” _kaggle.com_. https://www.kaggle.com/code/madz2000/pneumonia-detection-using-cnn-92-6-accuracy
# Team Member Contributions
  * Each team member should individually fill out the team member contribution MS form. https://forms.office.com/r/9HSrYJNvik (you will need to login to your GT credentials to complete this form, one response per student).

