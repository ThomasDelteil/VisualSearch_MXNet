FROM awsdeeplearningteam/mms_cpu:latest

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/nmslib/hnsw \
    && cd hnsw  \
    && pip install pybind11 numpy setuptools \
    && cd python_bindings \
    && python setup.py install



ENV PATH="/mxnet_model_server:${PATH}" 
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV OMP_NUM_THREADS=1

# Because of timeouts and issues in fargate, make a BIG image with everything included
COPY index.idx /data/visualsearch/mms/index.idx
COPY idx_ASIN.pkl /data/visualsearch/mms/idx_ASIN.pkl
COPY ASIN_data.pkl /data/visualsearch/mms/ASIN_data.pkl

COPY mms_app_cpu.conf /mxnet_model_server/

COPY visualsearch.model /mxnet_model_server/

LABEL maintainer="tdelteil@amazon.com"

CMD ["mxnet-model-server", "start", "--mms-config", "/mxnet_model_server/mms_app_cpu.conf"]