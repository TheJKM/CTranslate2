#include "./utils.h"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ctranslate2/utils.h"

#include "env.h"

namespace ctranslate2 {
  namespace cuda {

    const char* cublasGetStatusName(hipblasStatus_t status)
    {
      switch (status)
      {
      case HIPBLAS_STATUS_SUCCESS:
        return "HIPBLAS_STATUS_SUCCESS";
      case HIPBLAS_STATUS_NOT_INITIALIZED:
        return "HIPBLAS_STATUS_NOT_INITIALIZED";
      case HIPBLAS_STATUS_ALLOC_FAILED:
        return "HIPBLAS_STATUS_ALLOC_FAILED";
      case HIPBLAS_STATUS_INVALID_VALUE:
        return "HIPBLAS_STATUS_INVALID_VALUE";
      case HIPBLAS_STATUS_ARCH_MISMATCH:
        return "HIPBLAS_STATUS_ARCH_MISMATCH";
      case HIPBLAS_STATUS_MAPPING_ERROR:
        return "HIPBLAS_STATUS_MAPPING_ERROR";
      case HIPBLAS_STATUS_EXECUTION_FAILED:
        return "HIPBLAS_STATUS_EXECUTION_FAILED";
      case HIPBLAS_STATUS_INTERNAL_ERROR:
        return "HIPBLAS_STATUS_INTERNAL_ERROR";
      case HIPBLAS_STATUS_NOT_SUPPORTED:
        return "HIPBLAS_STATUS_NOT_SUPPORTED";
      case HIPBLAS_STATUS_UNKNOWN:
        return "HIPBLAS_STATUS_UNKNOWN";
      default:
        return "UNKNOWN";
      }
    }

    // We assign the default CUDA stream to the main thread since it can interact with
    // multiple devices (e.g. load replicas on each GPU). The main thread is created
    // before the others, so it will be the first to see the flag below set to true.
    static std::atomic<bool> is_main_thread(true);

    class CudaStream {
    public:
      CudaStream() {
        if (is_main_thread) {
          is_main_thread = false;
          _stream = hipStreamDefault;
        } else {
          CUDA_CHECK(hipGetDevice(&_device));
          CUDA_CHECK(hipStreamCreate(&_stream));
        }
      }
      ~CudaStream() {
        if (_stream != hipStreamDefault) {
          ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
          hipStreamDestroy(_stream);
        }
      }
      hipStream_t get() const {
        return _stream;
      }
    private:
      int _device;
      hipStream_t _stream;
    };

    class CublasHandle {
    public:
      CublasHandle() {
        CUDA_CHECK(hipGetDevice(&_device));
        CUBLAS_CHECK(hipblasCreate(&_handle));
        CUBLAS_CHECK(hipblasSetStream(_handle, get_cuda_stream()));
      }
      ~CublasHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        hipblasDestroy(_handle);
      }
      hipblasHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      hipblasHandle_t _handle;
    };

    // We create one cuBLAS/cuDNN handle per host thread. The handle is destroyed
    // when the thread exits.

    hipStream_t get_cuda_stream() {
      static thread_local CudaStream hip_stream;
      return hip_stream.get();
    }

    hipblasHandle_t get_cublas_handle() {
      static thread_local CublasHandle cublas_handle;
      return cublas_handle.get();
    }

#ifdef CT2_WITH_CUDNN
    class CudnnHandle {
    public:
      CudnnHandle() {
        CUDA_CHECK(hipGetDevice(&_device));
        CUDNN_CHECK(hipdnnCreate(&_handle));
        CUDNN_CHECK(hipdnnSetStream(_handle, get_cuda_stream()));
      }
      ~CudnnHandle() {
        ScopedDeviceSetter scoped_device_setter(Device::CUDA, _device);
        hipdnnDestroy(_handle);
      }
      hipdnnHandle_t get() const {
        return _handle;
      }
    private:
      int _device;
      hipdnnHandle_t _handle;
    };

    hipdnnHandle_t get_cudnn_handle() {
      static thread_local CudnnHandle cudnn_handle;
      return cudnn_handle.get();
    }

    hipdnnDataType_t get_cudnn_data_type(DataType dtype) {
      switch (dtype) {
      case DataType::FLOAT32:
        return HIPDNN_DATA_FLOAT;
      case DataType::FLOAT16:
        return HIPDNN_DATA_HALF;
      case DataType::BFLOAT16:
        return CUDNN_DATA_BFLOAT16;
      case DataType::INT32:
        return HIPDNN_DATA_INT32;
      case DataType::INT8:
        return HIPDNN_DATA_INT8;
      default:
        throw std::invalid_argument("No cuDNN data type for type " + dtype_name(dtype));
      }
    }
#endif

    int get_gpu_count() {
      int gpu_count = 0;
      hipError_t status = hipGetDeviceCount(&gpu_count);
      if (status != hipSuccess)
        return 0;
      return gpu_count;
    }

    bool has_gpu() {
      return get_gpu_count() > 0;
    }

    const hipDeviceProp_t& get_device_properties(int device) {
      static thread_local std::vector<std::unique_ptr<hipDeviceProp_t>> cache;

      if (device < 0) {
        CUDA_CHECK(hipGetDevice(&device));
      }
      if (device >= static_cast<int>(cache.size())) {
        cache.resize(device + 1);
      }

      auto& device_prop = cache[device];
      if (!device_prop) {
        device_prop = std::make_unique<hipDeviceProp_t>();
        CUDA_CHECK(hipGetDeviceProperties(device_prop.get(), device));
      }
      return *device_prop;
    }

    // See docs.nvidia.com/deeplearning/sdk/tensorrt-support-matrix/index.html
    // for hardware support of reduced precision.

    bool gpu_supports_int8(int device) {
      const hipDeviceProp_t& device_prop = get_device_properties(device);
      return device_prop.major > 6 || (device_prop.major == 6 && device_prop.minor == 1);
    }

    bool gpu_has_int8_tensor_cores(int device) {
      const hipDeviceProp_t& device_prop = get_device_properties(device);
      return device_prop.major > 7 || (device_prop.major == 7 && device_prop.minor >= 2);
    }

    bool gpu_has_fp16_tensor_cores(int device) {
      const hipDeviceProp_t& device_prop = get_device_properties(device);
      return device_prop.major >= 7;
    }

    bool have_same_compute_capability(const std::vector<int>& devices) {
      if (devices.size() > 1) {
        int ref_major = -1;
        int ref_minor = -1;
        for (const int device : devices) {
          const hipDeviceProp_t& device_prop = get_device_properties(device);
          const int major = device_prop.major;
          const int minor = device_prop.minor;
          if (ref_major < 0) {
            ref_major = major;
            ref_minor = minor;
          } else if (major != ref_major || minor != ref_minor)
            return false;
        }
      }

      return true;
    }

    static thread_local bool true_fp16_gemm = read_bool_from_env("CT2_CUDA_TRUE_FP16_GEMM", true);

    bool use_true_fp16_gemm() {
      return true_fp16_gemm;
    }

    void use_true_fp16_gemm(bool use) {
      true_fp16_gemm = use;
    }

  }
}
