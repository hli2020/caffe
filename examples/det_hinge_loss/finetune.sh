#!/usr/bin/env sh
# Hongyang LI
# action recognition version

caffe_path=/home/hongyang/research_office/caffe_act_reg/build/tools
mpi_path=/home/dependency/mpi_cuda/bin

# phase 1
# all layers except the last softmax (1001 output) have zero lr
#nohup 
GLOG_logtostderr=1 \
	$mpi_path/mpirun -np 2 \
	$caffe_path/caffe train \
	-weights=vgg_ft_iter_117500.caffemodel \
   	-solver=solver_1.prototxt \
2>&1 | tee hell.log  	
#> hinge_loss_1.log 2>&1 &

# # phase 2
# nohup $mpi_path/mpirun -np 1 \
# 	$caffe_path/caffe train \
# 	-weights=result/xxx.caffemodel \
#    	-solver=solver_2.prototxt \
#    	> hinge_loss_2.log 2>&1 &

# # phase 1
# nohup $mpi_path/mpirun -np 1 \
# 	$caffe_path/caffe train \
# 	-weights=result/xxx.caffemodel \
#    	-solver=solver_3.prototxt \
#    	> hinge_loss_3.log 2>&1 &
