#ifndef HAPI_BLAS_H
#define HAPI_BLAS_H

// The dense BLAS vendor table for the benchmark applications, in the style of the runtime's hapi_portable.h:
// cuBLAS on a CUDA build, hipBLAS on a HIP build, under one set of names. hipBLAS mirrors the cuBLAS API (handle,
// stream, workspace, math mode, op enums, sgemm), so every entry is a plain rename. An application that uses this
// links the vendor library itself (-lcublas / -lhipblas); the runtime does not.
//
// Why not KokkosBlas: its cuBLAS/rocBLAS path keeps ONE handle per process, sets the stream on it per call and gives
// no control over the workspace or math mode, while moe needs a handle per lane with a fixed workspace (its checksums
// require every GEMM of a given shape to pick the same algorithm) and runs experts on several PEs concurrently.

#include "hapi_portable.h"

#if CMK_HIP

#include <hipblas/hipblas.h>
typedef hipblasHandle_t hapiBlasHandle_t;
typedef hipblasStatus_t hapiBlasStatus_t;
#define hapiBlasStatusSuccess HIPBLAS_STATUS_SUCCESS
typedef hipblasOperation_t hapiBlasOperation_t;
#define HAPI_BLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define HAPI_BLAS_OP_N HIPBLAS_OP_N
#define HAPI_BLAS_OP_T HIPBLAS_OP_T
#define hapiBlasCreate hipblasCreate
#define hapiBlasDestroy hipblasDestroy
#define hapiBlasSetStream hipblasSetStream
#define hapiBlasSetWorkspace hipblasSetWorkspace
#define hapiBlasSgemm hipblasSgemm
// TF32 tensor-op math is a request the vendor may not honour; hipBLAS exposes the same switch where the hardware has
// it (MI300) and ignores it elsewhere.
#ifdef HIPBLAS_TF32_TENSOR_OP_MATH
static inline hapiBlasStatus_t hapiBlasSetMathModeTf32(hapiBlasHandle_t h, bool tf32) {
  return hipblasSetMathMode(h, tf32 ? HIPBLAS_TF32_TENSOR_OP_MATH : HIPBLAS_DEFAULT_MATH);
}
#else
static inline hapiBlasStatus_t hapiBlasSetMathModeTf32(hapiBlasHandle_t, bool) { return HIPBLAS_STATUS_SUCCESS; }
#endif

#else  /* CUDA */

#include <cublas_v2.h>
typedef cublasHandle_t hapiBlasHandle_t;
typedef cublasStatus_t hapiBlasStatus_t;
#define hapiBlasStatusSuccess CUBLAS_STATUS_SUCCESS
typedef cublasOperation_t hapiBlasOperation_t;
#define HAPI_BLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define HAPI_BLAS_OP_N CUBLAS_OP_N
#define HAPI_BLAS_OP_T CUBLAS_OP_T
#define hapiBlasCreate cublasCreate
#define hapiBlasDestroy cublasDestroy
#define hapiBlasSetStream cublasSetStream
#define hapiBlasSetWorkspace cublasSetWorkspace
#define hapiBlasSgemm cublasSgemm
static inline hapiBlasStatus_t hapiBlasSetMathModeTf32(hapiBlasHandle_t h, bool tf32) {
  return cublasSetMathMode(h, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
}

#endif

#endif
