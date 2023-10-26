#include <stdexcept>

#include <hipblas.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#ifdef _WIN32
#  include <windows.h>
#  define CUBLAS_LIBNAME "cublas64_" STR(CUBLAS_VER_MAJOR) ".dll"
#else
#  include <dlfcn.h>
#  define CUBLAS_LIBNAME "libcublas.so." STR(CUBLAS_VER_MAJOR)
#endif

#include <spdlog/spdlog.h>

#include "env.h"

namespace ctranslate2 {

  template <typename Signature>
  static Signature load_symbol(void* handle, const char* name, const char* library_name) {
#ifdef _WIN32
    void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
    void* symbol = dlsym(handle, name);
#endif
    if (!symbol)
      throw std::runtime_error("Cannot load symbol " + std::string(name)
                               + " from library " + std::string(library_name));
    return reinterpret_cast<Signature>(symbol);
  }

  static inline void log_cublas_version(void* handle) {
    using Signature = hipblasStatus_t(*)(libraryPropertyType, int*);
    const auto cublas_get_property = load_symbol<Signature>(handle,
                                                            "cublasGetProperty",
                                                            CUBLAS_LIBNAME);

    int major_version = 0;
    int minor_version = 0;
    int patch_level = 0;

    cublas_get_property(MAJOR_VERSION, &major_version);
    cublas_get_property(MINOR_VERSION, &minor_version);
    cublas_get_property(PATCH_LEVEL, &patch_level);

    spdlog::info("Loaded cuBLAS library version {}.{}.{}",
                 major_version, minor_version, patch_level);
  }

  static void* get_so_handle() {
    static auto so_handle = []() {
#ifdef _WIN32
      std::string cuda_path = read_string_from_env("CUDA_PATH");
      if (!cuda_path.empty()) {
        cuda_path += "\\bin";
        SetDllDirectoryA(cuda_path.c_str());
      }
      void* handle = static_cast<void*>(LoadLibraryA(CUBLAS_LIBNAME));
#else
      void* handle = dlopen(CUBLAS_LIBNAME, RTLD_LAZY);
#endif
      if (!handle)
        throw std::runtime_error("Library " + std::string(CUBLAS_LIBNAME)
                                 + " is not found or cannot be loaded");
      log_cublas_version(handle);
      return handle;
    }();
    return so_handle;
  }

  template <typename Signature>
  static Signature load_symbol(const char* name) {
    void* handle = get_so_handle();
    return load_symbol<Signature>(handle, name, CUBLAS_LIBNAME);
  }

}

// TODO: these stub functions could be automatically generated.

extern "C" {

  hipblasStatus_t hipblasCreate(hipblasHandle_t *handle) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t*);
    static auto func = ctranslate2::load_symbol<Signature>("hipblasCreate");
    return func(handle);
  }

  hipblasStatus_t hipblasDestroy(hipblasHandle_t handle) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t);
    static auto func = ctranslate2::load_symbol<Signature>("hipblasDestroy");
    return func(handle);
  }

  hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t stream) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t, hipStream_t);
    static auto func = ctranslate2::load_symbol<Signature>("hipblasSetStream");
    return func(handle, stream);
  }

  hipblasStatus_t cublasGetMathMode(hipblasHandle_t handle, cublasMath_t *mode) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t, cublasMath_t*);
    static auto func = ctranslate2::load_symbol<Signature>("cublasGetMathMode");
    return func(handle, mode);
  }

  hipblasStatus_t hipblasSgemm(hipblasHandle_t handle,
                                hipblasOperation_t transa,
                                hipblasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const float *alpha,
                                const float *A,
                                int lda,
                                const float *B,
                                int ldb,
                                const float *beta,
                                float *C,
                                int ldc) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t,
                                        hipblasOperation_t,
                                        hipblasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const float*,
                                        const float*,
                                        int,
                                        const float*,
                                        int,
                                        const float*,
                                        float*,
                                        int);
    static auto func = ctranslate2::load_symbol<Signature>("hipblasSgemm");
    return func(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t handle,
                                           hipblasOperation_t transa,
                                           hipblasOperation_t transb,
                                           int m,
                                           int n,
                                           int k,
                                           const float *alpha,
                                           const float *A,
                                           int lda,
                                           long long int strideA,
                                           const float *B,
                                           int ldb,
                                           long long int strideB,
                                           const float *beta,
                                           float *C,
                                           int ldc,
                                           long long int strideC,
                                           int batchCount) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t,
                                        hipblasOperation_t,
                                        hipblasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const float*,
                                        const float*,
                                        int,
                                        long long int,
                                        const float*,
                                        int,
                                        long long int,
                                        const float*,
                                        float*,
                                        int,
                                        long long int,
                                        int);
    static auto func = ctranslate2::load_symbol<Signature>("hipblasSgemmStridedBatched");
    return func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
  }

  hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle,
                              hipblasOperation_t transa,
                              hipblasOperation_t transb,
                              int m,
                              int n,
                              int k,
                              const void *alpha,
                              const void *A,
                              hipblasDatatype_t Atype,
                              int lda,
                              const void *B,
                              hipblasDatatype_t Btype,
                              int ldb,
                              const void *beta,
                              void *C,
                              hipblasDatatype_t Ctype,
                              int ldc,
                              hipblasDatatype_t computeType,
                              hipblasGemmAlgo_t algo) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t,
                                        hipblasOperation_t,
                                        hipblasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const void*,
                                        const void*,
                                        hipblasDatatype_t,
                                        int,
                                        const void*,
                                        hipblasDatatype_t,
                                        int,
                                        const void*,
                                        void*,
                                        hipblasDatatype_t,
                                        int,
                                        hipblasDatatype_t,
                                        hipblasGemmAlgo_t);
    static auto func = ctranslate2::load_symbol<Signature>("hipblasGemmEx");
    return func(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
  }

  hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t handle,
                                            hipblasOperation_t transa,
                                            hipblasOperation_t transb,
                                            int m,
                                            int n,
                                            int k,
                                            const void *alpha,
                                            const void *A,
                                            hipblasDatatype_t Atype,
                                            int lda,
                                            long long int strideA,
                                            const void *B,
                                            hipblasDatatype_t Btype,
                                            int ldb,
                                            long long int strideB,
                                            const void *beta,
                                            void *C,
                                            hipblasDatatype_t Ctype,
                                            int ldc,
                                            long long int strideC,
                                            int batchCount,
                                            hipblasDatatype_t computeType,
                                            hipblasGemmAlgo_t algo) {
    using Signature = hipblasStatus_t(*)(hipblasHandle_t,
                                        hipblasOperation_t,
                                        hipblasOperation_t,
                                        int,
                                        int,
                                        int,
                                        const void*,
                                        const void*,
                                        hipblasDatatype_t,
                                        int,
                                        long long int,
                                        const void*,
                                        hipblasDatatype_t,
                                        int,
                                        long long int,
                                        const void*,
                                        void*,
                                        hipblasDatatype_t,
                                        int,
                                        long long int,
                                        int,
                                        hipblasDatatype_t,
                                        hipblasGemmAlgo_t);
    static auto func = ctranslate2::load_symbol<Signature>("hipblasGemmStridedBatchedEx");
    return func(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
  }

}
