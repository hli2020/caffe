#!/usr/bin/env sh

# hli2020_ultimate on MacBook
CAFFE_PATH=/Users/leefrancis/Documents/github/caffe_6_1/build/tools
mpi_path=/path/to/your/mpi

GLOG_logtostderr=1 $mpi_path/mpirun -np 1 $CAFFE_PATH/caffe time \
		-model=/your/architecture_model.prototxt \
        -gpu=0
        2>&1 | tee log/alexnet_v4_debug.log


