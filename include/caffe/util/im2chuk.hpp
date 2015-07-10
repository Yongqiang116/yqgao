#ifndef _CAFFE_UTIL_IM2CHUK_HPP_
#define _CAFFE_UTIL_IM2CHUK_HPP_

namespace caffe {

template <typename Dtype>
void im2chuk_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void chuk2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, Dtype* data_im);

template <typename Dtype>
void im2chuk_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void chuk2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int local_h, const int local_w, const int stride_h,
    const int stride_w, Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2CHUK_HPP_
