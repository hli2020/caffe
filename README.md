# Multi-bias Activation

This is the code repository (`bias' branch) for the paper "Multi-Bias Non-linear Activation in Deep Neural Networks" published in the 33rd International Conference on Machine Learning (ICML), New York, NY, USA, 2016. [arXiv](https://arxiv.org/abs/1604.00676), [official paper](http://jmlr.org/proceedings/papers/v48/lia16.pdf). These two versions share little difference in contents.

Check out the `demo.sh' for running the bias module. We show an example on the CIFAR-10 dataset, which achieves 6.73% (Table 4 in the paper) without data augmentation.

The slides of the talk at the conference is available [here](http://www.ee.cuhk.edu.hk/~yangli/icml16_bias.pdf).

## License and Citation

The code is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

Please cite our work if it helps your research:

	@inproceedings{li2016bias,
	  Author = {Li, Hongyang and Ouyang, Wanli and Wang, Xiaogang},
	  Booktitle = {ICML},
	  Title = {Multi-Bias Non-linear Activation in Deep Neural Networks},
	  Year = {2016}
	}
