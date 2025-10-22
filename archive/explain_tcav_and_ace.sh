source /home/stu13/s8/ia3494/miniforge3/etc/profile.d/conda.sh
conda deactivate && conda activate py38

export CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CUDA_HOME/include:$CPATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

CUDA_VISIBLE_DEVICES=6 python explainer_tcav_and_ace.py