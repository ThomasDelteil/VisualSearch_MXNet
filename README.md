
# Visual Search with MXNet Gluon and 1M Amazon Product images

Pre-requisite:
- MXNet: `pip install --pre mxnet-cu91`
- hnswlib (follow the guide here: https://github.com/nmslib/hnsw)


```python
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import multiprocessing
from mxnet.gluon.data.vision.datasets import ImageFolderDataset
from mxnet.gluon.data import DataLoader
import numpy as np
import wget
import json
import pickle
import hnswlib
import numpy as np
import glob, os, time
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import urllib.parse
import urllib
import gzip
%matplotlib inline
```

Data originally from here:
http://jmcauley.ucsd.edu/data/amazon/

*Image-based recommendations on styles and substitutes
J. McAuley, C. Targett, J. Shi, A. van den Hengel
SIGIR, 2015*

## Downloading images
We only use a subset of the total number of images, here 1M
(it takes about 40 minutes to download all the data on an ec2 instance)


```python
subset_num = 1000000
```

Beware, if using the full dataset this will download **300GB** of images, make sure you have the appropriate hardware and connexion!
Alternatively, just set `images_path` to a directory containing images following this format `ID.jpg`


```python
data_path = 'metadata.json'
images_path = '/data/amazon_images_subset'
```


```python
num_lines = 0
num_lines = sum(1 for line in open(data_path))
assert num_lines >= subset_num, "Subset needs to be smaller or equal to total number of example"
```

Download the metadata.json file that contains the URL of the images


```python
if not os.path.isfile(data_path):
    # Downloading the metadata, 3.1GB, unzipped 9GB
    !wget -nv https://s3.us-east-2.amazonaws.com/mxnet-public/stanford_amazon/metadata.json.gz
    !gzip -d metadata.json.gz

if not os.path.isdir(images_path):
    os.makedirs(images_path)
```


```python
def parse(path, num_cpu, modulo):
    g = open(path, 'r')
    for i, l in enumerate(g):
        if (i >= num_lines - subset_num and i%num_cpu == modulo):
            yield eval(l)
```


```python
def download_files(modulo):
    for data in parse(data_path, NUM_CPU, modulo):
        if 'imUrl' in data and data['imUrl'] is not None and 'categories' in data and data['imUrl'].split('.')[-1] == 'jpg':
            url = data['imUrl']
            try:
                path = os.path.join(images_path, data['asin']+'.jpg')
                if not os.path.isfile(path):
                    file = urllib.request.urlretrieve(url, path)
            except:
                print("Error downloading {}".format(url))
```

Downloading the images using 10 times more processes than cores


```python
NUM_CPU = multiprocessing.cpu_count()*10
```


```python
pool = multiprocessing.Pool(processes=NUM_CPU)
pool.map(download_files, list(range(NUM_CPU)))
```

## Generate the image embedding


```python
BATCH_SIZE = 256
EMBEDDING_SIZE = 512
SIZE = (224, 224)
MEAN_IMAGE= mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])
```

### Featurizer
We use a pre-trained model from the model zoo


```python
ctx = mx.gpu()
```


```python
net = vision.resnet18_v2(pretrained=True, ctx=ctx)
net = net.features
```

### Data Transform
to convert the images to a shape usable by the network


```python
def transform(image, label):
    resized = mx.image.resize_short(image, SIZE[0]).astype('float32')
    cropped, crop_info = mx.image.center_crop(resized, SIZE)
    cropped /= 255.
    normalized = mx.image.color_normalize(cropped,
                                      mean=MEAN_IMAGE,
                                      std=STD_IMAGE) 
    transposed = nd.transpose(normalized, (2,0,1))
    return transposed, label
```

### Load the data


```python
import os, tempfile, glob
empty_folder = tempfile.mkdtemp()
# Create an empty image Folder Data Set
dataset = ImageFolderDataset(root=empty_folder, transform=transform)
```


```python
list_files = glob.glob(os.path.join(images_path, '**.jpg'))
```

We take a of the files for speed sake, and replace the items of the ImageFolderDataset


```python
#uncomment to use only a subset of the files
#subset_num = 1000000
#list_files = list_files[-subset_num:]
dataset.items = list(zip(list_files, [0]*len(list_files)))
```

We load the dataset in a dataloader with as many workers as CPU cores


```python
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, last_batch='keep', shuffle=False, num_workers=multiprocessing.cpu_count())
```

### Run the data through the featurizer


```python
features = np.zeros((len(dataset), EMBEDDING_SIZE), dtype=np.float32)
```


```python
%%time
tick = time.time()
n_print = 100
j = 0
for i, (data, label) in enumerate(dataloader):
    data = data.as_in_context(ctx)
    if i%n_print == 0 and i > 0:
        print("{0} batches, {1} images, {2:.3f} img/sec".format(i, i*BATCH_SIZE, BATCH_SIZE*n_print/(time.time()-tick)))
        tick = time.time()
    output = net(data)
    features[(i)*BATCH_SIZE:(i+1)*max(BATCH_SIZE, len(output)), :] = output.asnumpy().squeeze()
```

## Create the search index


```python
# Number of elements in the index
num_elements = len(features)
labels_index = np.arange(num_elements)
```


```python
%%time 

# Declaring index
p = hnswlib.Index(space = 'l2', dim = EMBEDDING_SIZE) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 100, M = 16)

# Element insertion (can be called several times):
int_labels = p.add_items(features, labels_index)

# Controlling the recall by setting ef:
p.set_ef(100) # ef should always be > k
```


```python
p.save_index('index.idx')
```

### Testing

We test the results by sampling random images from the dataset and searching their K-NN


```python
def plot_predictions(images):
    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(15, 15))
    gs.update(hspace=0.1, wspace=0.1)
    for i, (gg, image) in enumerate(zip(gs, images)):
        gg2 = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=gg)
        ax = fig.add_subplot(gg2[:,:])
        ax.imshow(image, cmap='Greys_r')
        ax.tick_params(axis='both',       
                       which='both',      
                       bottom='off',      
                       top='off',         
                       left='off',
                       right='off',
                       labelleft='off',
                       labelbottom='off') 
        ax.axes.set_title("result [{}]".format(i))
        if i == 0:
            plt.setp(ax.spines.values(), color='red')
            ax.axes.set_title("SEARCH".format(i))
```


```python
def search(N, k):
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    q_labels, q_distances = p.knn_query([features[N]], k = k)
    images = [plt.imread(dataset.items[label][0]) for label in q_labels[0]]
    plot_predictions(images)
```

### Random testing


```python
%%time
index = np.random.randint(0,len(features))
k = 6
search(index, k)
```

    CPU times: user 196 ms, sys: 32 ms, total: 228 ms
    Wall time: 216 ms



![png](images/test_1.png)


### Manual testing


```python
path = 'skirt.jpg'
```


```python
p.set_ef(300) # ef should always be > k
image = plt.imread(path)[:,:,:3]
image_t, _ = transform(nd.array(image), 1)
output = net(image_t.expand_dims(axis=0).as_in_context(ctx))
labels, distances = p.knn_query([output.asnumpy().reshape(-1,)], k = 5)
images = [image]
images += [plt.imread(dataset.items[label][0]) for label in labels[0]]
```


```python
plot_predictions(images)
```


![png](images/test_2.png)

