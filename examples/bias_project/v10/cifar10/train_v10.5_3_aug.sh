#!/usr/bin/env sh
#	--snapshot=result_8.3/cifar100_iter_15000.solverstate \ 		
#  	--weights=../benchmark/result/cifar10_quick_iter_7000.caffemodel \
#CAFFE_PATH=/home/hongyang/research_office/caffe_history/build/tools
#mpi_path=/home/hongyang/research_office/mpi185/bin

# ls139
CAFFE_PATH=/media/DATADISK/hyli/project/caffe_bias/build/tools
mpi_path=/home/hyli/dependency/openmpi/bin
# s190
#CAFFE_PATH=/data2/project/caffe_bias/build/tools
#mpi_path=/home/software/mpi_cuda/bin

log_name=v10.5_3_aug

GLOG_logtostderr=1 \
		$mpi_path/mpirun -np 1 \
		$CAFFE_PATH/caffe train \
  		--solver=solver/solver_$log_name.prototxt \
		--gpu=0 \
2>&1 | tee log/cifar10_$log_name.log

sh ../../tools/sorted_log/parse_log_cifar.sh log/cifar10_$log_name.log 
mv cifar10_$log_name.log.* ../../tools/sorted_log/  
