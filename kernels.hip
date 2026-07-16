#include "kernels.h"

__device__ float bl_sample_texture(const hipTextureObject_t *txt, float x, float y) {

}

__device__ float bl_sample_array(const float *img, unsigned w, unsigned h, float x, float y) {
    float x0 = floor(x);
    float y0 = floor(y);
    float alpha = x-x0;
    float beta = y-y0;
    unsigned i = static_cast<unsigned>(x0) + w*static_cast<unsigned>(y0);
    unsigned i_x = i + (x<w? 1: 0);
    unsigned i_y = i + (y<h? w: 0);
    unsigned i_xy = i + (x<w? 1: 0);
    return beta*(alpha*img[i_xy]+(1-alpha)*img[y]) + (1-beta)*(alpha*img[i_xy+(1-alpha)*img[i]);
}

struct alignas(16) pixel_delta {
    float value, dx, dy, dxy;
};

__device__ float bl_sample_precomp(const pixel_delta *pcp, unsigned w, unsighed h, float x, float y) {
    float x0 = floor(x);
    float y0 = floor(y);
    float alpha = x-x0;
    float beta = y-y0;
    unsigned i = static_cast<unsigned>(x0) + w*static_cast<unsigned>(y0);
    return pxp[i]+alpha*pxp[i+1]+beta*pxp[i+2]+alpha*beta*pxp[i+3];
}

__global__ make_pixel_delta_array(const float *img, unsigned w, unsigned h, pixel_data *pcp) {
    unsigned pxl_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned pxl_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (pxl_x>=w || pxl_y>=h) return;

    unsigned pxl_i = pxl_x + pxl_y*w;
    unsigned pxl_i_x = pxl_i + (pxl_x==w-1? 0: 1);
    unsigned pxl_i_y = pxl_i + (pxl_y==h-1? 0: w);
    unsigned pxl_i_xy = pxl_i_y + (pxl_x==w-1? 0: 1);

    pcp[pxl_i] = { img[pxl_i], img[pxl_i_x]-img[pxl_i], img[pxl_i_y]-img[pxl_i], img[pxl_i_xy]-img[pxl_i] };
}

// one thread per output pixel
__global__ subsample_image_array(const float *src, unsigned src_w, unsigned src_h, float *out, unsigned out_w, unsigned out_h) {
    unsigned pxl_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned pxl_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (pxl_x>=w || pxl_y>=h) return;

    float src_x = (pxl_x*(float(out_w))/out_w;
    float src_y = (pxl_y*(float(out_h))/out_h;

    out[pxl_x+out_w*pxl_y] = bl_sample_array(src, src_w, src_h, src_x, src_y);
}

__global__ subsample_pixel_delta_array(const pixel_delta *src, unsigned src_w, unsigned src_h, float *out, unsigned out_w, unsigned out_h) {
    unsigned pxl_x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned pxl_y = threadIdx.y + blockIdx.y*blockDim.y;

    if (pxl_x>=w || pxl_y>=h) return;

    float src_x = (pxl_x*(float(out_w))/out_w;
    float src_y = (pxl_y*(float(out_h))/out_h;

    out[pxl_x+out_w*pxl_y] = bl_sample_precomp(src, src_w, src_h, src_x, src_y);
}

void run_make_pixel_delta_array(const float *img, unsigned w, unsigned h, pixel_data *pcp) {
    dim2 block(8, 8);
    dim2 grid((out_w+block.x-1)/block.x, (out_h+block.y-1)/block.y);
    make_pixel_delta_array<<<grid, block>>>(img, w, h, pcp);
}

void run_subsample_image_array(const float *src, unsigned src_w, unsigned src_h, float *out, unsigned out_w, unsigned out_h) {
    dim2 block(8, 8);
    dim2 grid((out_w+block.x-1)/block.x, (out_h+block.y-1)/block.y);
    subsample_image_array<<<grid, block>>>(src, src_w, src_h, out, out_w, out_h);
}

void run_subsample_pixel_delta_array(const pixel_delta *src, unsigned src_w, unsigned src_h, float *out, unsigned out_w, unsigned out_h) {
    dim2 block(8, 8);
    dim2 grid((out_w+block.x-1)/block.x, (out_h+block.y-1)/block.y);
    subsample_image_array<<<grid, block>>>(src, src_w, src_h, out, out_w, out_h);
}
