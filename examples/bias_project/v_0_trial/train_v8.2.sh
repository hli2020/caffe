#!/usr/bin/env sh

#  	--weights=../benchmark/result/cifar10_quick_iter_7000.caffemodel \
#CAFFE_PATH=/home/hongyang/research_office/caffe_history/build/tools
#mpi_path=/home/hongyang/research_office/mpi185/bin

# ls139
CAFFE_PATH=/media/DATADISK/hyli/project/caffe_bias/build/tools
mpi_path=/home/hyli/dependency/openmpi/bin

GLOG_logtostderr=1 \
		$mpi_path/mpirun -np 2 \
		$CAFFE_PATH/caffe train \
  		--solver=solver_v8.2.prototxt \
  		--gpu=0,1 \
2>&1 | tee cifar_v8.2_gpu=2.log 


  

