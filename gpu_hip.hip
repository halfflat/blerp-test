#include <hip/hip_runtime.h>
#include "gpu.h"
#include "gpu_hip.h"

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

gpuError_t gpuMallocPitch(void** devPtr, std::size_t *pitch, std::size_t width, std::size_t height) {
    return static_cast<gpuError_t>(hipMallocPitch(devPtr, pitch, width, height));
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

gpuError_t gpuCreateTextureObjectFromSingleChannelFloatImage(gpuTextureObject_t* tobj_ptr, const void* dev_image_data, std::size_t img_w, std::size_t img_h, std::size_t img_byte_pitch) {
    hipResourceDesc res_desc;

    res_desc.resType = hipResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = const_cast<void*>(dev_image_data);
    res_desc.res.pitch2D.desc = hipCreateChannelDesc<float>();
    res_desc.res.pitch2D.height = img_h;
    res_desc.res.pitch2D.pitchInBytes = img_byte_pitch;
    res_desc.res.pitch2D.width = img_w;

    hipTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));

    tex_desc.addressMode[0] = hipAddressModeBorder;
    tex_desc.addressMode[1] = hipAddressModeBorder;
    tex_desc.addressMode[2] = hipAddressModeBorder;
    tex_desc.filterMode = hipFilterModePoint;
    tex_desc.readMode = hipReadModeElementType;
    tex_desc.normalizedCoords = 0;

    hipTextureObject_t hip_tobj;
    auto err = static_cast<gpuError_t>(hipCreateTextureObject(&hip_tobj, &res_desc, &tex_desc, nullptr));
    tobj_ptr->actual_gpu_texture_object_impl = hip_tobj;
    return static_cast<gpuError_t>(err);
}

gpuError_t gpuDestroyTextureObject(gpuTextureObject_t tobj) {
    hipTextureObject_t hip_tobj = static_cast<hipTextureObject_t>(tobj.actual_gpu_texture_object_impl);
    return static_cast<gpuError_t>(hipDestroyTextureObject(hip_tobj));
}

