#pragma(once)

struct alignas(16) pixel_delta {
    float value, dx, dy, dxy;
};


void run_make_pixel_delta_array(const float *img, unsigned w, unsigned h, pixel_delta *pcp);

void run_subsample_image_array(const float *src, unsigned src_w, unsigned src_h, float *out, unsigned out_w, unsigned out_h);

void run_subsample_pixel_delta_array(const pixel_delta *src, unsigned src_w, unsigned src_h, float *out, unsigned out_w, unsigned out_h);
