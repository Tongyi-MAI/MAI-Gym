// Modified by MAI-Gym contributors.
#ifndef __CUDA_PROFILER_API_H__
#define __CUDA_PROFILER_API_H__


#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

extern __host__ cudaError_t CUDARTAPI cudaProfilerStart(void);
extern __host__ cudaError_t CUDARTAPI cudaProfilerStop(void);

#if defined(__cplusplus)
}
#endif

#endif
