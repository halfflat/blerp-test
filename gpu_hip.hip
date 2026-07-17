#include <hip/hip_runtime.h>
#include "gpu.h"

#pragma(once)

#include <cstddef>

const char* gpuGetErrorString(gpuError_t err) {
    return hipGetErrorString(static_cast<hipError_t>(err));
}

gpuError_t gpuInit(unsigned flags) {
    return static_cast<gpuError_t>(hipInit(flags));
}

gpuError_t gpuMalloc(void** devPtr, std::size_t size) {
    return static_cast<gpuError_t>(hipMalloc(devPtr, size));
}

gpuError_t gpuMemcpyHtoD(void *dst, const void* src, std::size_t count) {
    return static_cast<gpuError_t>(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));
}
gpuError_t gpuMemcpyDtoH(void *dst, const void* src, std::size_t count) {
    return static_cast<gpuError_t>(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));
}

gpuError_t gpuEventCreate(gpuEvent_t* evptr) {
    hipEvent_t hip_ev;
    auto err = hipEventCreate(&hip_ev);
    evptr->actual_gpu_event_impl = hip_ev;
    return static_cast<gpuError_t>(err);
}

gpuError_t gpuEventRecord(gpuEvent_t ev, gpuStream_t stream) {
    hipEvent_t hip_ev = static_cast<hipEvent_t>(ev.actual_gpu_event_impl);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream.actual_gpu_stream_impl);
    return static_cast<gpuError_t>(hipEventRecord(hip_ev, hip_stream));
}

gpuError_t gpuEventSynchronize(gpuEvent_t ev) {
    hipEvent_t hip_ev = static_cast<hipEvent_t>(ev.actual_gpu_event_impl);
    return static_cast<gpuError_t>(hipEventSynchronize(hip_ev));
}

gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end) {
    hipEvent_t hip_ev1 = static_cast<hipEvent_t>(start.actual_gpu_event_impl);
    hipEvent_t hip_ev2 = static_cast<hipEvent_t>(end.actual_gpu_event_impl);
    return static_cast<gpuError_t>(hipEventElapsedTime(ms, hip_ev1, hip_ev2));
}

