#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2chuk.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2chuk_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w,
    const int stride_h, const int stride_w,
    Dtype* data_chuk) {
  int height_chuk = kernel_h + stride_h * (local_h - 1);
  int width_chuk = kernel_w + stride_w * (local_w - 1);
  int num_h_chuk = (height - height_chuk) / stride_h / local_h + 1;
  int num_w_chuk = (width - width_chuk) / stride_w / local_w + 1;    
  int channels_chuk = channels * num_h_chuk * num_w_chuk;
  for (int c = 0; c < channels_chuk; ++c) {
    int c_th_chuk = c / channels;
    int c_im = c % channels;
    int h_offset = c_th_chuk / num_w_chuk;
    int w_offset = c_th_chuk % num_w_chuk;
    for (int h = 0; h < height_chuk; ++h) {
      for (int w = 0; w < width_chuk; ++w) {
        int h_im = h_offset * stride_h * local_h + h;
        int w_im = w_offset * stride_w * local_w + w;
        data_chuk[(c * height_chuk + h) * width_chuk + w] =
          data_im[(c_im * height + h_im) * width + w_im];          
      }
    }
  }
}

// Explicit instantiation
template void im2chuk_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, float* data_chuk);
template void im2chuk_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, double* data_chuk);

template <typename Dtype>
void chuk2im_cpu(const Dtype* data_chuk, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_chuk = kernel_h + stride_h * (local_h - 1);
  int width_chuk = kernel_w + stride_w * (local_w - 1);
  int num_h_chuk = (height - height_chuk) / stride_h / local_h + 1;
  int num_w_chuk = (width - width_chuk) / stride_w / local_w + 1;    
  int channels_chuk = channels * num_h_chuk * num_w_chuk;
  for (int c = 0; c < channels_chuk; ++c) {
    int c_th_chuk = c / channels;
    int c_im = c % channels;
    int h_offset = c_th_chuk / num_w_chuk;
    int w_offset = c_th_chuk % num_w_chuk;
    for (int h = 0; h < height_chuk; ++h) {
      for (int w = 0; w < width_chuk; ++w) {
        int h_im = h_offset * stride_h * local_h + h;
        int w_im = w_offset * stride_w * local_w + w;
          data_im[(c_im * height + h_im) * width + w_im] = 
            data_chuk[(c * height_chuk + h) * width_chuk + w];
      }
    }
  }
}

// Explicit instantiation
template void chuk2im_cpu<float>(const float* data_chuk, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, float* data_im);
template void chuk2im_cpu<double>(const double* data_chuk, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, double* data_im);

}  // namespace caffe
