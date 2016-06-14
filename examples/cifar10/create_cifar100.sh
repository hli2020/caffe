#!/usr/bin/env sh
# This script converts the cifar data into leveldb format.

EXAMPLE=examples/cifar10
DATA=data/cifar-100-binary
DBTYPE=lmdb

echo "Creating $DBTYPE..."

#rm -rf $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/cifar10_test_$DBTYPE

./build/examples/cifar10/convert_cifar100_data.bin $DATA $EXAMPLE $DBTYPE

#echo "Computing image mean..."

#./build/tools/compute_image_mean -backend=$DBTYPE \
#  $EXAMPLE/cifar10_train_$DBTYPE $EXAMPLE/mean.binaryproto

echo "Done."
