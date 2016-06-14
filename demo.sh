#!/usr/bin/env sh

CAFFE_PATH=build/tools
mpi_path=/path/to/mpi/bin

log_name=cifar10_no_aug

GLOG_logtostderr=1 \
		$mpi_path/mpirun -np 2 \
		$CAFFE_PATH/caffe train \
  		--solver=solver.prototxt \
		--gpu=0,1 \
2>&1 | tee $log_name.log



  

