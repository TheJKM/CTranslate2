#pragma once

#include <string>

#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <thrust/execution_policy.h>

#ifdef CT2_WITH_CUDNN
#  include <hipDNN.h>
#endif

#include "ctranslate2/types.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace cuda {

#define CUDA_CHECK(ans)                                                 \
    {                                                                   \
      hipError_t code = (ans);                                         \
      if (code != hipSuccess)                                          \
        THROW_RUNTIME_ERROR("CUDA failed with error "                   \
                            + std::string(hipGetErrorString(code)));   \
    }

#define CUBLAS_CHECK(ans)                                               \
    {                                                                   \
      hipblasStatus_t status = (ans);                                    \
      if (status != HIPBLAS_STATUS_SUCCESS)                              \
        THROW_RUNTIME_ERROR("cuBLAS failed with status "                \
                            + std::string(ctranslate2::cuda::cublasGetStatusName(status))); \
    }

#define CUDNN_CHECK(ans)                                                \
    {                                                                   \
      hipdnnStatus_t status = (ans);                                     \
      if (status != HIPDNN_STATUS_SUCCESS)                               \
        THROW_RUNTIME_ERROR("cuDNN failed with status "                 \
                            + std::string(hipdnnGetErrorString(status))); \
    }

    const char* cublasGetStatusName(hipblasStatus_t status);

    hipStream_t get_cuda_stream();
    hipblasHandle_t get_cublas_handle();
#ifdef CT2_WITH_CUDNN
    hipdnnHandle_t get_cudnn_handle();
    hipdnnDataType_t get_cudnn_data_type(DataType dtype);
#endif

    int get_gpu_count();
    bool has_gpu();
    const hipDeviceProp_t& get_device_properties(int device = -1);
    bool gpu_supports_int8(int device = -1);
    bool gpu_has_int8_tensor_cores(int device = -1);
    bool gpu_has_fp16_tensor_cores(int device = -1);
    bool have_same_compute_capability(const std::vector<int>& devices);

    bool use_true_fp16_gemm();
    void use_true_fp16_gemm(bool use);

    class UseTrueFp16GemmInScope {
    public:
      UseTrueFp16GemmInScope(const bool use)
        : _previous_value(use_true_fp16_gemm())
        , _scope_value(use)
      {
        use_true_fp16_gemm(_scope_value);
      }

      ~UseTrueFp16GemmInScope() {
        use_true_fp16_gemm(_previous_value);
      }

    private:
      const bool _previous_value;
      const bool _scope_value;
    };

// Convenience macro to call Thrust functions with a default execution policy.
#define THRUST_CALL(FUN, ...) FUN(thrust::hip::par_nosync.on(ctranslate2::cuda::get_cuda_stream()), __VA_ARGS__)

  }
}
