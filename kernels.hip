#include <hip/hip_runtime.h>
#include "gpu_hip.h"
#include "kernels.h"

__device__ float bl_sample_array(const float *img, unsigned stride, unsigned w, unsigned h, float x, float y) {
    float x0 = floor(x);
    float y0 = floor(y);
    float alpha = x-x0;
    float beta = y-y0;
    unsigned i = static_cast<unsigned>(x0) + stride*static_cast<unsigned>(y0);
    unsigned i_x = i + (x+1<w? 1: 0);
    unsigned i_y = i + (y+1<h? stride: 0);
    unsigned i_xy = i_y + (x+1<w? 1: 0);
    return beta*(alpha*img[i_xy]+(1-alpha)*img[i_y]) + (1-beta)*(alpha*img[i_x]+(1-alpha)*img[i]);
}

__device__ float bl_sample_texture(gpuTextureObject_t texobj, unsigned w, unsigned h, float x, float y) {
    float x0 = floor(x);
    float y0 = floor(y);
    float alpha = x-x0;
    float beta = y-y0;

    float t00 = gpuTex2D<float>(texobj, x0, y0);
    float t10 = gpuTex2D<float>(texobj, x0+1, y0);
    float t01 = gpuTex2D<float>(texobj, x0, y0+1);
    float t11 = gpuTex2D<float>(texobj, x0+1, y0+1);
    return beta*(alpha*t11+(1-alpha)*t01) + (1-beta)*(alpha*t10+(1-alpha)*t00);
}


__device__ float bl_sample_precomp(const pixel_delta *pxd, unsigned w, unsigned h, float x, float y) {
    float x0 = floor(x);
    float y0 = floor(y);
    float alpha = x-x0;
    float beta = y-y0;
    unsigned i = static_cast<unsigned>(x0) + w*static_cast<unsigned>(y0);
    return pxd[i].value+alpha*pxd[i].dx+beta*pxd[i].dy+alpha*beta*pxd[i].dxy;
}

__global__ void make_pixel_delta_array(const float *img, unsigned stride, unsigned w, unsigned h, pixel_delta *pxd) {
    unsigned pxl_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned pxl_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (pxl_x>=w || pxl_y>=h) return;

    unsigned pxl_i = pxl_x + pxl_y*w;
    unsigned pxl_i_x = pxl_i + (pxl_x==w-1? 0: 1);
    unsigned pxl_i_y = pxl_i + (pxl_y==h-1? 0: w);
    unsigned pxl_i_xy = pxl_i_y + (pxl_x==w-1? 0: 1);

    unsigned img_i = pxl_x + pxl_y*stride;
    unsigned img_i_x = pxl_i + (pxl_x==w-1? 0: 1);
    unsigned img_i_y = pxl_i + (pxl_y==h-1? 0: stride);
    unsigned img_i_xy = pxl_i_y + (pxl_x==w-1? 0: 1);

    pxd[pxl_i] = { img[img_i], img[img_i_x]-img[img_i], img[img_i_y]-img[img_i], img[img_i_xy]-img[img_i] };
}

// one thread per output pixel
__global__ void subsample_image_array(const float *src, unsigned src_stride, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h) {
    unsigned pxl_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned pxl_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (pxl_x>=out_w || pxl_y>=out_h) return;

    float src_x = (pxl_x*float(src_w))/out_w;
    float src_y = (pxl_y*float(src_h))/out_h;

    out[pxl_x+out_stride*pxl_y] = bl_sample_array(src, src_stride, src_w, src_h, src_x, src_y);
}

__global__ void subsample_texture(gpuTextureObject_t texobj, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h) {
    unsigned pxl_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned pxl_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (pxl_x>=out_w || pxl_y>=out_h) return;

    float src_x = (pxl_x*float(src_w))/out_w;
    float src_y = (pxl_y*float(src_h))/out_h;

    out[pxl_x+out_stride*pxl_y] = bl_sample_texture(texobj, src_w, src_h, src_x, src_y);
}

__global__ void subsample_pixel_delta_array(const pixel_delta *src, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h) {
    unsigned pxl_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned pxl_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (pxl_x>=out_w || pxl_y>=out_h) return;

    float src_x = (pxl_x*float(src_w))/out_w;
    float src_y = (pxl_y*float(src_h))/out_h;

    out[pxl_x+out_stride*pxl_y] = bl_sample_precomp(src, src_w, src_h, src_x, src_y);
}

void run_make_pixel_delta_array(const float *img, unsigned stride, unsigned w, unsigned h, pixel_delta *pcp) {
    dim3 block(8, 8);
    dim3 grid((w+block.x-1)/block.x, (h+block.y-1)/block.y);
    make_pixel_delta_array<<<grid, block>>>(img, stride, w, h, pcp);
}

void run_subsample_image_array(const float *src, unsigned src_stride, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h) {
    dim3 block(8, 8);
    dim3 grid((out_w+block.x-1)/block.x, (out_h+block.y-1)/block.y);
    subsample_image_array<<<grid, block>>>(src, src_stride, src_w, src_h, out, out_stride, out_w, out_h);
}

void run_subsample_pixel_delta_array(const pixel_delta *src, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h) {
    dim3 block(8, 8);
    dim3 grid((out_w+block.x-1)/block.x, (out_h+block.y-1)/block.y);
    subsample_pixel_delta_array<<<grid, block>>>(src,  src_w, src_h, out, out_stride, out_w, out_h);
}

void run_subsample_texture(gpuTextureObject_t texobj, unsigned src_w, unsigned src_h, float *out, unsigned out_stride, unsigned out_w, unsigned out_h) {
    dim3 block(8, 8);
    dim3 grid((out_w+block.x-1)/block.x, (out_h+block.y-1)/block.y);
    subsample_texture<<<grid, block>>>(texobj, src_w, src_h, out, out_stride, out_w, out_h);
}

gpuError_t make_texture_object(const float *img, unsigned stride, unsigned w, unsigned h, gpuTextureObject_t *texobj_ptr) {
    std::size_t byte_pitch = stride*sizeof(float);
    return gpuCreateTextureObjectFromSingleChannelFloatImage(texobj_ptr, static_cast<const void*>(img), w, h, byte_pitch);
}
