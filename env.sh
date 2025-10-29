export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
