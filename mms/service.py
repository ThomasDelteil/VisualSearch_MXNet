from mms.model_service.mxnet_model_service import MXNetBaseService
from mxnet.gluon.data import ArrayDataset
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
import base64
from PIL import Image
import io
import json
import pickle
import logging
import os 

# External dependencies
import hnswlib

# Load the network
ctx = mx.cpu()

# Fixed parameters
SIZE = (224, 224)
MEAN_IMAGE= mx.nd.array([0.485, 0.456, 0.406])
STD_IMAGE = mx.nd.array([0.229, 0.224, 0.225])
EMBEDDING_SIZE = 512
INDEX_URL = 'https://s3.us-east-2.amazonaws.com/mxnet-public/visual_search/index.idx'
IDX_ASIN_URL = 'https://s3.us-east-2.amazonaws.com/mxnet-public/visual_search/idx_ASIN.pkl'
ASIN_DATA_URL = 'https://s3.us-east-2.amazonaws.com/mxnet-public/visual_search/ASIN_data.pkl'
EF = 300
K = 25

# Data Transform
def transform(image):
    resized = mx.image.resize_short(image, SIZE[0]).astype('float32')
    cropped, crop_info = mx.image.center_crop(resized, SIZE)
    cropped /= 255.
    normalized = mx.image.color_normalize(cropped,
                                      mean=MEAN_IMAGE,
                                      std=STD_IMAGE) 
    transposed = nd.transpose(normalized, (2,0,1))
    return transposed

class VisualSearchService(MXNetBaseService):
    
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(VisualSearchService, self).__init__(model_name, model_dir, manifest, gpu)
        
        ############################################
        logging.info('Downloading Resources Files')
        
        data_dir = os.environ.get('DATA_DIR', '/data/visualsearch/mms/')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        index_url = os.environ.get('INDEX_URL', INDEX_URL)
        idx_to_ASIN_url = os.environ.get('IDX_ASIN_URL', IDX_ASIN_URL)
        ASIN_to_data_url = os.environ.get('ASIN_DATA_URL', ASIN_DATA_URL)

        mx.test_utils.download(index_url, dirname=data_dir)
        mx.test_utils.download(idx_to_ASIN_url, dirname=data_dir)
        mx.test_utils.download(ASIN_to_data_url, dirname=data_dir)
        ############################################
        
        ############################################
        logging.info('Loading Resources files')
        
        self.idx_ASIN = pickle.load(open(os.path.join(data_dir, 'idx_ASIN.pkl'), 'rb'))
        self.ASIN_data = pickle.load(open(os.path.join(data_dir,'ASIN_data.pkl'), 'rb'))        
        self.p = hnswlib.Index(space = 'l2', dim = EMBEDDING_SIZE)
        self.p.load_index(os.path.join(data_dir,'index.idx'))
        ############################################
        
        logging.info('Resources files loaded')
        
        
        self.p.set_ef(EF)        
        self.k = K
        
    def _preprocess(self, data):
        image_b64 = data[0][0]
        image_b64 = image_b64[22:] if image_b64[:4] == 'data' else image_b64
        image_bytes = base64.b64decode(str.encode(image_b64))
        image_PIL = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image_PIL)
        image_t = transform(nd.array(image_np[:, :, :3]))
        image_batchified = image_t.expand_dims(axis=0).as_in_context(ctx)
        return [image_batchified]
    
    def _postprocess(self, data):
        labels, distances = self.p.knn_query([data[0].asnumpy().reshape(-1,)], k = self.k)
        logging.info(labels)
        output = []
        for label in labels[0]:
            ASIN = self.idx_ASIN[label]
            output.append(self.ASIN_data[ASIN])
        return output