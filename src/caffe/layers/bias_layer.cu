#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// CUDA kernel for forward
template <typename Dtype>
__global__ void BiasForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* bias_data, const int div_factor) {

  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] + bias_data[c];
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void BiasBackward(const int n, const Dtype* in_diff, Dtype* out_diff) {

  CUDA_KERNEL_LOOP(index, n) {
    
    out_diff[index] = in_diff[index];
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void BiasParamBackward(const int n, const Dtype* in_diff, Dtype* out_diff) {

  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index];
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* bias_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // NOLINT_NEXT_LINE(whitespace/operators)
  BiasForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, bias_data, div_factor);
  CUDA_POST_KERNEL_CHECK;

  //LOG(INFO) << "gpu:: bias layer";
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // Propagate to param
  if (this->param_propagate_down_[0]) {

  	Dtype* bias_diff = this->blobs_[0]->mutable_gpu_diff();
  	// bias_diff is set as 0, then accumulated over batches
  	//caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), bias_diff);
  	
    int cdim = channels * dim;
  	Dtype dsum = 0.;

  	for (int n = 0; n < bottom[0]->num(); ++n) {

  		Dtype* temp_buff = multiplier_.mutable_gpu_diff();

  		// compute element-wise diff
  		// NOLINT_NEXT_LINE(whitespace/operators)
  		BiasParamBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  			cdim, top_diff + top[0]->offset(n), multiplier_.mutable_gpu_diff());
  		CUDA_POST_KERNEL_CHECK;

  		// I do not have a single clue about what the hell happens here
  		if (channel_shared_) {
  			Dtype d;
  			caffe_gpu_dot<Dtype>(channels * dim, multiplier_.gpu_diff(), multiplier_.gpu_data(), &d);
  			dsum += d;

  		} else {
  			caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
            multiplier_.gpu_diff(), multiplier_.gpu_data(), 1., bias_diff);

  		}

  	}	//end for loop

  	if (channel_shared_) {
  		//caffe_gpu_set(this->blobs_[0]->count(), Dtype(dsum), bias_diff);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), bias_diff);
  	}

  }

  // Propagate to bottom
  if (propagate_down[0]) {

  	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  	// NOLINT_NEXT_LINE(whitespace/operators)
  	BiasBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  		count, top_diff, bottom_diff);

  	CUDA_POST_KERNEL_CHECK;
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(BiasLayer);

} 	// namespace caffe
