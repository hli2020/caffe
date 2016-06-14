#!/usr/bin/env sh
#	--snapshot=result_8.3/cifar100_iter_15000.solverstate \ 		
#  	--weights=../benchmark/result/cifar10_quick_iter_7000.caffemodel \
#CAFFE_PATH=/home/hongyang/research_office/caffe_history/build/tools
#mpi_path=/home/hongyang/research_office/mpi185/bin

# ls139
#CAFFE_PATH=/media/DATADISK/hyli/project/caffe_bias/build/tools
#mpi_path=/home/hyli/dependency/openmpi/bin
# s190
CAFFE_PATH=/data2/project/caffe_bias/build/tools
mpi_path=/home/software/mpi_cuda/bin
# ls149
#CAFFE_PATH=/home/hyli/project/caffe_bias/build/tools
#mpi_path=/home/hyli/dependency/openmpi/bin

GLOG_logtostderr=1 \
		$mpi_path/mpirun -np 4 \
		$CAFFE_PATH/caffe train \
  		--solver=solver/solver_v9.4.prototxt \
		--gpu=0,1,2,3 \
2>&1 | tee log/cifar10_v9.4_gpu=4.log 


  

