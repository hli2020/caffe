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

log_name=v10.6_1

GLOG_logtostderr=1 \
		$mpi_path/mpirun -np 2 \
		$CAFFE_PATH/caffe train \
  		--solver=solver/solver_$log_name.prototxt \
		--snapshot=result/cifar10_v10.6_1_iter_87000.solverstate \
		--gpu=6,7 \
2>&1 | tee log/cifar10_r1_$log_name.log

		echo "tidy up results..." 
		sh ../../tools/sorted_log/parse_log_cifar.sh log/cifar10_r1_$log_name.log 
		echo "done!" 
  

