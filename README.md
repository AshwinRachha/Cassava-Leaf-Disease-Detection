# Cassava-Leaf-Disease-Detection

Cassava is the second largest provider of carbohydrates in the continent of Africa and is key to growth of small farmers because it can withstand harsh conditions. A major percentage of farmers rely on growing this crop but its vulnerability to viral diseases is a major impediment for growth. Image recognition is a fast and inexpensive means of screening the crops for any present viral disease and can offer means to mitigate the spread of the disease by take effective and immediate remedial actions. Transfer learning has proven to be one of the most effective methods in various fields of computer vision in recent times. In this kernel we fine tune an EfficientNet based model to achieve fair results on the Cassava Leaf Disease Classification Dataset. The program is written in fastai.

```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

Now we will need to import all the necessary libraries with some additional functionality. The code below will import all of the libraries necessary to run the code.


```python
import numpy as np
import pandas as pd
import os
from fastai.vision import *
from fastai.vision.all import *
from fastai.callback import *
```


```python
import warnings
warnings.filterwarnings('ignore')
```

With the rise of transfer learning, the essentiality of scaling has been deeply realised for enhancing the performance as well as efficieny of models. Traditionaly scaling can be done in three dimensions viz. depth, width and resolution in terms of convolutional neural networks. Depth scaling pertains to increasing the number of layers in the model, making it more deeper; width scaling makes the model wider (one possible way is to increase the number of channels in a layer) and resolution scaling means using high resolution images so that features are more fine-grained. Each method applied individually has some drawbacks such as in depth scaling we have the problem of vanishing gradients and in width scaling the accuracy saturates after a point and there is a limit to increasing resolution of images and a slight increase doesnt result in significant improvement of performance. Hence Efficientnets are proposed to deal with balancing all dimensions of a network during CNN scaling for getting improved accuracy and efficieny. The authors proposed a simple yet very effective scaling technique which uses a compound coefficientto uniformly scale network width, depth, and resolution in a principled way. We used the pytorch wrapper for efficientnets. To install run the following command:  


```python
pip install efficientnet_pytorch
```

```python
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
```


```python
data_root = Path('../input/cassava-leaf-disease-classification')
os.listdir(data_root)
```




    ['sample_submission.csv',
     'train_images',
     'test_tfrecords',
     'train_tfrecords',
     'label_num_to_disease_map.json',
     'test_images',
     'train.csv']




```python
train = pd.read_csv(data_root / 'train.csv')
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000015157.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000201771.jpg</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100042118.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000723321.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000812911.jpg</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Since Fastai dataloaders need to know the entire filepath of an image from a dataframe, we have to replace the image_id i.e name of the image in the train dataframe to the entire path of the image. Images for training are kept in the train_images folder and the public test images are kept in the test_images folder.


```python
train['img_path'] = train['image_id'].map(lambda x : data_root/'train_images'/x)
train = train.drop(columns = ['image_id'])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>img_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>../input/cassava-leaf-disease-classification/train_images/1000015157.jpg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>../input/cassava-leaf-disease-classification/train_images/1000201771.jpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>../input/cassava-leaf-disease-classification/train_images/100042118.jpg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>../input/cassava-leaf-disease-classification/train_images/1000723321.jpg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>../input/cassava-leaf-disease-classification/train_images/1000812911.jpg</td>
    </tr>
  </tbody>
</table>
</div>



Now we visualize a sample Image from the training set.


```python
from PIL import Image
img = Image.open(train['img_path'][44])
img
```

![Test](/blog/img/efficientnet-b4-integrated-with-fastai%20(2)_14_0.png "Test")


Now we come to perform some data augmentation on our images for better prediction. The item_tfms picks a random scaled crop of an image and resizes it to the size mentioned. Next we perform batch transformations that performs a series of augmentations such as resizing the images to 224X224, flipping, rotating, zooming and affine transformations.


```python
item_tfms = RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))
batch_tfms = [*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)]
```

Here we load the data from our training dataframe using the ImageDataLoader class which is the highest level of API for loading data. We pass all the required parameters into the dataloader.


```python
data = ImageDataLoaders.from_df(train, 
                               valid_pct=0.2, 
                               seed=999, 
                               label_col=0, 
                               fn_col=1, 
                               bs=32, 
                               item_tfms=item_tfms,
                               batch_tfms=batch_tfms) 
```



```python
data.c
```




    5



Here we can visualize a batch of images with their related labels pertaining to the type of disease.


```python
data.show_batch(figsize = (15,15))
```


![Test](/blog/img/efficientnet-b4-integrated-with-fastai%20(2)_23_0.png "Test")


Lets train a simple efficientnet-b4 model.


```python
model = EfficientNet.from_pretrained('efficientnet-b4')
```

Before we train our model lets define a metric that gives additional perpective regarding its performance. We define the Top 3 accuracy metric which tells us the measure of how often our predicted class falls in the top 3 values of our softmax distribution. In this case we have 5 classes, each with a respective softmax distribution. Top3 accuracy is calculated by the number of the images whose labels fall in the top 3 classes of the predicted softmax distribution out of all the predictions made.


```python
top3 = partial(top_k_accuracy, k=3)
```

Now we define the prebuilt learner class which lets us use transfer learning models rapidly without much hassle. For the loss function we have used SOTA label smoothing crossentropy which shows promising results in image classification. We can also used mixed precision easily. 


```python
learn = Learner(data, model, metrics=[accuracy, top3], loss_func=LabelSmoothingCrossEntropy()).to_fp16()
```

 In order to train a model, we need to find the most optimal learning rate, which can be done with fastai's learning rate finder:


```python
learn.unfreeze()
learn.lr_find()

```








    SuggestedLRs(lr_min=0.025118863582611083, lr_steep=0.0014454397605732083)



![Test](/blog/img/efficientnet-b4-integrated-with-fastai%20(2)_31_2.png "Test")


As show above the optimal learning rate to be used with out model should be just before the loss takes a steep dip. Here we settle with 1e-2 for the learning rate. 


```python
learn.fine_tune(10,base_lr=1e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>top_k_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.776684</td>
      <td>6.929300</td>
      <td>0.395419</td>
      <td>0.515307</td>
      <td>06:49</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>top_k_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.508496</td>
      <td>1.515002</td>
      <td>0.819117</td>
      <td>0.965412</td>
      <td>06:33</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.526023</td>
      <td>1.629592</td>
      <td>0.771208</td>
      <td>0.952559</td>
      <td>06:30</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.507683</td>
      <td>1.867913</td>
      <td>0.741295</td>
      <td>0.940874</td>
      <td>06:37</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.496514</td>
      <td>4.045783</td>
      <td>0.301239</td>
      <td>0.888292</td>
      <td>06:42</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.498794</td>
      <td>1.673173</td>
      <td>0.753447</td>
      <td>0.940173</td>
      <td>06:46</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.407032</td>
      <td>1.510584</td>
      <td>0.819584</td>
      <td>0.968918</td>
      <td>06:49</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.376257</td>
      <td>1.414135</td>
      <td>0.857443</td>
      <td>0.979668</td>
      <td>06:50</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.358680</td>
      <td>1.382066</td>
      <td>0.861416</td>
      <td>0.980136</td>
      <td>06:47</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.322549</td>
      <td>1.363431</td>
      <td>0.869596</td>
      <td>0.982239</td>
      <td>06:46</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.304314</td>
      <td>1.364084</td>
      <td>0.869829</td>
      <td>0.982473</td>
      <td>06:43</td>
    </tr>
  </tbody>
</table>


* Voila! With just a few lines, we applied SOTA techniques to achieve a good accuracy and top3accuracy. Thats not bad.


```python
learn.recorder.plot_loss()
```

![Test](/blog/img/efficientnet-b4-integrated-with-fastai (2)_35_0.png "Test")


Fastai provides an easy functionality to plot the confusion matrix. This matrix gives an indication of actually predicted classes and the classes which belong to one class but have been misclassified as another. The diagonal pertains to all the images that have been rightly classified.


```python
interp = ClassificationInterpretation.from_learner(learn)
```






```python
interp.plot_confusion_matrix()
```

![Test](/blog/img/efficientnet-b4-integrated-with-fastai%20(2)_38_0.png "Test")


Now we will make our submissions


```python
sample_df = pd.read_csv(data_root/'sample_submission.csv')
sample_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2216849948.jpg</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample_df1 = sample_df.copy()
sample_df1['img_path'] = sample_df1['image_id'].map(lambda x:data_root/'test_images'/x)
sample_df1 = sample_df1.drop(columns=['image_id'])
test_dl = data.test_dl(sample_df1)
```


```python
test_dl.show_batch()
```


Now lets get the predictions. We will use the common technique known as the Test time Augmentation prebuilt to get predictions. 


```python
preds, _ = learn.tta(dl=test_dl, beta=0)
```

```python
sample_df['label'] = preds.argmax(dim=-1).numpy()
```


```python
sample_df.to_csv('submission.csv',index=False)
```

And we are done!

