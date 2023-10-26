#include "ctranslate2/ops/conv1d.h"

#include "cuda/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Conv1D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output) const {
#ifndef CT2_WITH_CUDNN
      (void)input;
      (void)weight;
      (void)bias;
      (void)output;
      throw std::runtime_error("Conv1D on GPU currently requires the cuDNN library "
                               "which is not integrated in this build");

#else
      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int kernel_size = weight.dim(2);

      hipdnnDataType_t data_type = cuda::get_cudnn_data_type(input.dtype());

      hipdnnTensorDescriptor_t input_desc;
      CUDNN_CHECK(hipdnnCreateTensorDescriptor(&input_desc));
      CUDNN_CHECK(hipdnnSetTensor4dDescriptor(input_desc, HIPDNN_TENSOR_NCHW, data_type,
                                             batch_size, in_channels, 1, input_length));

      hipdnnTensorDescriptor_t output_desc;
      CUDNN_CHECK(hipdnnCreateTensorDescriptor(&output_desc));
      CUDNN_CHECK(hipdnnSetTensor4dDescriptor(output_desc, HIPDNN_TENSOR_NCHW, data_type,
                                             batch_size, out_channels, 1, output_length));

      hipdnnFilterDescriptor_t weight_desc;
      CUDNN_CHECK(hipdnnCreateFilterDescriptor(&weight_desc));
      CUDNN_CHECK(hipdnnSetFilter4dDescriptor(weight_desc, data_type, HIPDNN_TENSOR_NCHW,
                                             out_channels, in_channels, 1, kernel_size));

      hipdnnConvolutionDescriptor_t conv_desc;
      CUDNN_CHECK(hipdnnCreateConvolutionDescriptor(&conv_desc));
      CUDNN_CHECK(hipdnnSetConvolution2dDescriptor(conv_desc,
                                                  /*pad_h=*/0, /*pad_w=*/_padding,
                                                  /*stride_h=*/1, /*stride_w=*/_stride,
                                                  /*dilation_h=*/1, /*dilation_w=*/_dilation,
                                                  HIPDNN_CROSS_CORRELATION,
                                                  HIPDNN_DATA_FLOAT));

      CUDNN_CHECK(hipdnnSetConvolutionMathType(conv_desc, HIPDNN_DEFAULT_MATH));
      if (data_type == HIPDNN_DATA_HALF)
        CUDNN_CHECK(hipdnnSetConvolutionMathType(conv_desc, HIPDNN_TENSOR_OP_MATH));

      hipdnnHandle_t handle = cuda::get_cudnn_handle();

      hipdnnConvolutionFwdAlgo_t algo = (bias
                                        ? HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                                        : HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);

      size_t workspace_size = 0;
      void* workspace = nullptr;
      CUDNN_CHECK(hipdnnGetConvolutionForwardWorkspaceSize(handle,
                                                          input_desc,
                                                          weight_desc,
                                                          conv_desc,
                                                          output_desc,
                                                          algo,
                                                          &workspace_size));

      if (workspace_size > 0)
        workspace = get_allocator<Device::CUDA>().allocate(workspace_size);

      float alpha = 1;
      float beta = 0;

      if (bias) {
        hipdnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(hipdnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(hipdnnSetTensor4dDescriptor(bias_desc, HIPDNN_TENSOR_NCHW, data_type,
                                               1, out_channels, 1, 1));

        hipdnnActivationDescriptor_t activation_desc;
        CUDNN_CHECK(hipdnnCreateActivationDescriptor(&activation_desc));
        CUDNN_CHECK(hipdnnSetActivationDescriptor(activation_desc,
                                                 HIPDNN_ACTIVATION_PATHTRU,
                                                 HIPDNN_NOT_PROPAGATE_NAN,
                                                 /*coef=*/0));

        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(handle,
                                                          &alpha,
                                                          input_desc,
                                                          input.buffer(),
                                                          weight_desc,
                                                          weight.buffer(),
                                                          conv_desc,
                                                          algo,
                                                          workspace,
                                                          workspace_size,
                                                          &beta,
                                                          output_desc,
                                                          output.buffer(),
                                                          bias_desc,
                                                          bias->buffer(),
                                                          activation_desc,
                                                          output_desc,
                                                          output.buffer()));

        CUDNN_CHECK(hipdnnDestroyActivationDescriptor(activation_desc));
        CUDNN_CHECK(hipdnnDestroyTensorDescriptor(bias_desc));

      } else {
        CUDNN_CHECK(hipdnnConvolutionForward(handle,
                                            &alpha,
                                            input_desc,
                                            input.buffer(),
                                            weight_desc,
                                            weight.buffer(),
                                            conv_desc,
                                            algo,
                                            workspace,
                                            workspace_size,
                                            &beta,
                                            output_desc,
                                            output.buffer()));
      }

      if (workspace)
        get_allocator<Device::CUDA>().free(workspace);

      CUDNN_CHECK(hipdnnDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CHECK(hipdnnDestroyFilterDescriptor(weight_desc));
      CUDNN_CHECK(hipdnnDestroyTensorDescriptor(input_desc));
      CUDNN_CHECK(hipdnnDestroyTensorDescriptor(output_desc));
#endif
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Conv1D::compute<Device::CUDA, T>(const StorageView& input,          \
                                     const StorageView& weight,         \
                                     const StorageView* bias,           \
                                     StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

  }
}
