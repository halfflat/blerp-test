#pragma(once)

#include "gpu.h"

struct alignas(16) pixel_delta {
    float value, dx, dy, dxy;
};


void run_make_pixel_delta_array(const float *img, unsigned stride, unsigned w, unsigned h, pixel_delta *pcp);

void run_subsample_image_array(const float *src, unsigned src_stride, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h);

void run_subsample_pixel_delta_array(const pixel_delta *src, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h);

gpuError_t make_texture_object(const float *img, unsigned stride, unsigned w, unsigned h, gpuTextureObject_t *texobj_ptr);

void run_subsample_texture(gpuTextureObject_t texobj, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h);
