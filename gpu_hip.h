#pragma once

#include <hip/hip_runtime.h>
#include "gpu.h"

template <typename T>
__device__ inline float gpuTex2D(gpuTextureObject_t obj, float u, float v) {
    hipTextureObject_t hip_tobj = static_cast<hipTextureObject_t>(obj.actual_gpu_texture_object_impl);
    return tex2D<T>(hip_tobj, u, v);
}

