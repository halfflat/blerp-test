#pragma once 

#include <cstddef>

// opaque ...
typedef enum gpuError_t { gpuSuccess = 0 } gpuError_t;

struct gpuEvent_t {
    void* actual_gpu_event_impl = nullptr;
};

struct gpuStream_t {
    void* actual_gpu_stream_impl = nullptr;
};

const char* gpuGetErrorString(gpuError_t);
gpuError_t gpuInit(unsigned flags);
gpuError_t gpuMalloc(void** devPtr, std::size_t size);
gpuError_t gpuMallocPitch(void** devPtr, std::size_t *byte_pitch, std::size_t width, std::size_t height);
gpuError_t gpuMemcpyHtoD(void *dst, const void* src, std::size_t count);
gpuError_t gpuMemcpyDtoH(void *dst, const void* src, std::size_t count);

gpuError_t gpuEventCreate(gpuEvent_t*);
gpuError_t gpuEventRecord(gpuEvent_t, gpuStream_t = gpuStream_t{});
gpuError_t gpuEventSynchronize(gpuEvent_t);
gpuError_t gpuEventElapsedTime(float* ms, gpuEvent_t start, gpuEvent_t end);

struct gpuTextureObject_t {
    void* actual_gpu_texture_object_impl = nullptr;
};

gpuError_t gpuCreateTextureObjectFromSingleChannelFloatImage(gpuTextureObject_t* tobj_ptr, const void* dev_image_data, std::size_t img_w, std::size_t img_h, std::size_t img_byte_pitch);
