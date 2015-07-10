#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2chuk.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2chuk_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int channels,
    const int stride_h, const int stride_w,
    const int height_chuk, const int width_chuk,
    Dtype* data_chuk) {
  CUDA_KERNEL_LOOP(index, n) {
    // int num_h_chuk = (height - height_chuk) / stride_h + 1;
    int num_w_chuk = (width - width_chuk) / stride_w + 1;
    int th_width = index / channels % num_w_chuk;
    int th_height = index / channels / num_w_chuk;
    int channel_in = index % channels;
    int height_in = th_height * stride_h;
    int width_in = th_width * stride_w;
    Dtype* data_chuk_ptr = data_chuk;
    data_chuk_ptr += (index * height_chuk + 0) * width_chuk + 0;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + height_in) * width + width_in;
    for (int i = 0; i < height_chuk; ++i) {
      for (int j = 0; j < width_chuk; ++j) {
        *(data_chuk_ptr + i * width_chuk + j) = 
          data_im_ptr[i * width + j];
      }
    }
  }
}

template <typename Dtype>
void im2chuk_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w,
    const int stride_h, const int stride_w,
    Dtype* data_chuk) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_chuk = kernel_h + stride_h * (local_h - 1);
  int width_chuk = kernel_w + stride_w * (local_w - 1);
  int num_h_chuk = (height - height_chuk) / stride_h / local_h + 1;
  int num_w_chuk = (width - width_chuk) / stride_w / local_w + 1; 

  int num_kernels = channels * num_h_chuk * num_w_chuk;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2chuk_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, channels,
      stride_h * local_h, stride_w * local_w, 
      height_chuk, width_chuk, data_chuk);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2chuk_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h, const int stride_w,
    float* data_chuk);
template void im2chuk_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    double* data_chuk);

template <typename Dtype>
__global__ void chuk2im_gpu_kernel(const int n, const Dtype* data_chuk,
    const int height, const int width, const int channels,
    const int stride_h, const int stride_w,
    const int height_chuk, const int width_chuk,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    // int num_h_chuk = (height - height_chuk) / stride_h + 1;
    int num_w_chuk = (width - width_chuk) / stride_w + 1;
    int th_width = index / channels % num_w_chuk;
    int th_height = index / channels / num_w_chuk;
    int channel_in = index % channels;
    int height_in = th_height * stride_h;
    int width_in = th_width * stride_w;
    const Dtype* data_chuk_ptr = data_chuk;
    data_chuk_ptr += (index * height_chuk + 0) * width_chuk + 0;
    Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + height_in) * width + width_in;
    for (int i = 0; i < height_chuk; ++i) {
      for (int j = 0; j < width_chuk; ++j) {
        *(data_im_ptr + i * width + j) = 
          data_chuk_ptr[i * width_chuk + j];
      }
    }
  }
}

template <typename Dtype>
void chuk2im_gpu(const Dtype* data_chuk, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, Dtype* data_im) {
  int height_chuk = kernel_h + stride_h * (local_h - 1);
  int width_chuk = kernel_w + stride_w * (local_w - 1);
  int num_h_chuk = (height - height_chuk) / stride_h / local_h + 1;
  int num_w_chuk = (width - width_chuk) / stride_w / local_w + 1; 

  int num_kernels = channels * num_h_chuk * num_w_chuk;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  chuk2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_chuk, height, width, channels,
      stride_h * local_h, stride_w * local_w, 
      height_chuk, width_chuk, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void chuk2im_gpu<float>(const float* data_chuk, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, float* data_im);
template void chuk2im_gpu<double>(const double* data_chuk, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, double* data_im);

}  // namespace caffe
