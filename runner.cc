#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <source_location>
#include <string>
#include <vector>

#include <tiffio.h>

#include "kernels.h"
#include "gpu.h"

using namespace std::literals;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;

struct fatal_error: std::runtime_error {
    fatal_error(const std::string& message): std::runtime_error(message) {}
};

void fatal(const std::string& msg = "") { throw fatal_error(msg); }

struct usage_error: std::runtime_error {
    usage_error(const std::string& message): std::runtime_error(message) {}
};

void usage(const std::string& msg = "") { throw usage_error(msg); }

void emit_usage_error(const std::string& msg) {
    if (!msg.empty()) std::cerr << "blerp-test: " << msg << '\n';
    std::cerr << "usage: blerp-test METHOD SAMPLING FILE [OUTFILE]\n\n    METHOD\t\tOne of array, texture, precomp.\n    SAMPLING\t\tDegree of oversampling in X and Y axes.\n    FILE\t\tTIFF file (greyscale) for sampling.\n"
        "    OUTFILE\t\t(Optional) write oversampled image to OUTFILE.\n\n";
}

void gpu_assert(gpuError_t err, std::source_location where = std::source_location::current()) {
    if (!err) return;
    fatal(std::format("GPU error line {}: {}", where.line(), gpuGetErrorString(err)));
}

template <typename R> void transform_scanline(const char* raw, float* out, uint32_t w) {
    double pxl_scale = 1;
    if constexpr (std::is_integral_v<R>) {
        pxl_scale = 1./std::numeric_limits<R>::max();
    }
    std::transform((const R*)raw, (const R*)raw+w, out, [pxl_scale](auto pxl) { return float(pxl*pxl_scale); });
}

int main(int argc, char** argv) {
    try {
        if (argc<4 || argc>5) usage();

        const char* out_filename = argv[4]; // nullptr if argc==4

        enum { method_array, method_texture, method_precomp } method;
        if (!strcmp("array", argv[1])) method = method_array;
        else if (!strcmp("texture", argv[1])) method = method_texture;
        else if (!strcmp("precomp", argv[1])) method = method_precomp;
        else usage();

        char* end = nullptr;
        double over = std::strtod(argv[2], &end);
        if (!end || *end) usage();

        const char* src_filename = argv[3];
        TIFF* tiff_src = TIFFOpen(src_filename, "r");
        if (!tiff_src) fatal(std::format("failed to open TIFF image '{}'", src_filename));

        uint32_t src_h = 0, src_w = 0;
        TIFFGetField(tiff_src, TIFFTAG_IMAGEWIDTH, &src_w);
        TIFFGetField(tiff_src, TIFFTAG_IMAGELENGTH, &src_h);
        if (src_h==0 || src_w==0) fatal("Failed to get image dimensions");

        uint16_t src_channels = 0, src_bpp = 0, src_fmt = 0;
        TIFFGetField(tiff_src, TIFFTAG_SAMPLESPERPIXEL, &src_channels);
        TIFFGetField(tiff_src, TIFFTAG_BITSPERSAMPLE, &src_bpp);
        TIFFGetFieldDefaulted(tiff_src, TIFFTAG_SAMPLEFORMAT, &src_fmt, uint16_t(1));
        if (src_channels!=1) fatal("Expected greyscale (single-channel) image");

        if (src_fmt!=SAMPLEFORMAT_UINT && src_fmt!=SAMPLEFORMAT_IEEEFP) fatal("unsupported TIFF sample format");

        uint16_t src_photometric = 0, src_orientation = 0;
        TIFFGetField(tiff_src, TIFFTAG_PHOTOMETRIC, &src_photometric);
        TIFFGetFieldDefaulted(tiff_src, TIFFTAG_ORIENTATION, &src_orientation, uint16_t(1));

        std::vector<float> src_data(src_h*src_w);
        float pxl_scale = float(1ull<<(src_bpp-1));

        auto scanline_size = TIFFScanlineSize(tiff_src);
        unsigned bytes_per_pixel = scanline_size/src_w;
        if ((bytes_per_pixel&(bytes_per_pixel-1))!=0 || bytes_per_pixel>4) fatal("can't work with this number of bytes per pixel in scanline");

        std::vector<char> scanline_raw(scanline_size);

        for (uint32_t row = 0; row<src_h; ++row) {
            if (TIFFReadScanline(tiff_src, &scanline_raw[0], row, 0)<0) fatal("TIFFReadScaline failed");

            if (src_fmt==SAMPLEFORMAT_IEEEFP)
                transform_scanline<float>(&scanline_raw[0], &src_data[row*src_w], src_w);
            else switch (bytes_per_pixel) {
            case 1:
                transform_scanline<uint8_t>(&scanline_raw[0], &src_data[row*src_w], src_w);
                break;
            case 2:
                transform_scanline<uint16_t>(&scanline_raw[0], &src_data[row*src_w], src_w);
                break;
            case 4:
                transform_scanline<uint32_t>(&scanline_raw[0], &src_data[row*src_w], src_w);
                break;
            default:
                fatal("internal logic error");
            }
        }

        uint32_t out_w = uint32_t(std::round(over*src_w));
        uint32_t out_h = uint32_t(std::round(over*src_h));
        std::vector<float> out_data(out_h*out_w);

        void* src_dev_ptr_raw = nullptr;
        void* out_dev_ptr_raw = nullptr;
        void* pxd_dev_ptr_raw = nullptr;

        gpu_assert(gpuInit(0));
        gpu_assert(gpuMalloc(&src_dev_ptr_raw, src_data.size()*sizeof(float)));
        gpu_assert(gpuMalloc(&out_dev_ptr_raw, out_data.size()*sizeof(float)));
        gpu_assert(gpuMalloc(&pxd_dev_ptr_raw, src_w*src_h*sizeof(pixel_delta)));
        gpu_assert(gpuMemcpyHtoD(src_dev_ptr_raw, &src_data[0], src_data.size()*sizeof(float)));

        gpuEvent_t ev_setup, ev_start, ev_stop;

        gpu_assert(gpuEventCreate(&ev_setup));
        gpu_assert(gpuEventCreate(&ev_start));
        gpu_assert(gpuEventCreate(&ev_stop));

        const float* src_dev_ptr = static_cast<const float*>(src_dev_ptr_raw);
        float* out_dev_ptr = static_cast<float*>(out_dev_ptr_raw);
        pixel_delta* pxd_dev_ptr = static_cast<pixel_delta*>(pxd_dev_ptr_raw);

        const unsigned n_iter = 1000;

        gpuEventRecord(ev_setup);
        switch (method) {
        case method_array:
            gpuEventRecord(ev_start);
            for (unsigned i = 0; i<n_iter; ++i) run_subsample_image_array(src_dev_ptr, src_w, src_h, out_dev_ptr, out_w, out_h);
            break;
        case method_precomp:
            run_make_pixel_delta_array(src_dev_ptr, src_w, src_h, pxd_dev_ptr);
            gpuEventRecord(ev_start);
            for (unsigned i = 0; i<n_iter; ++i) run_subsample_pixel_delta_array(pxd_dev_ptr, src_w, src_h, out_dev_ptr, out_w, out_h);
            break;
        case method_texture:
            fatal("unimplemented");
        }

        gpuEventRecord(ev_stop);
        gpu_assert(gpuEventSynchronize(ev_stop));

        float time_setup = 0, time_run = 0;
        gpuEventElapsedTime(&time_setup, ev_setup, ev_start);
        gpuEventElapsedTime(&time_run, ev_start, ev_stop);

        std::cout << std::format("normalized kernel execution times (ms) across {} invocations:\nsetup: {}\nsampling: {}\n", n_iter, time_setup, time_run/n_iter);

        gpu_assert(gpuMemcpyDtoH(&out_data[0], out_dev_ptr_raw, out_data.size()*sizeof(float)));
        if (out_filename) {
            TIFF* tiff_out = TIFFOpen(out_filename, "w");
            if (!tiff_out) fatal("failed to open '"s+out_filename+"' for writing");

            TIFFSetField(tiff_out, TIFFTAG_IMAGEWIDTH, out_w);
            TIFFSetField(tiff_out, TIFFTAG_IMAGELENGTH, out_h);
            TIFFSetField(tiff_out, TIFFTAG_BITSPERSAMPLE, 8);
            TIFFSetField(tiff_out, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(tiff_out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            TIFFSetField(tiff_out, TIFFTAG_PHOTOMETRIC, src_photometric);
            TIFFSetField(tiff_out, TIFFTAG_ORIENTATION, src_orientation);
            TIFFSetField(tiff_out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(tiff_out, TIFFTAG_ROWSPERSTRIP, out_h);

            std::vector<uint8_t> out_data_u8(out_h*out_w);
            std::transform(out_data.begin(), out_data.begin()+out_h*out_w, &out_data_u8[0], [](float p) { return uint8_t(255*std::clamp(p, 0.f, 1.f)); });

            auto n = TIFFWriteEncodedStrip(tiff_out, 0, &out_data_u8[0], out_w*out_h*sizeof(uint8_t));
            TIFFClose(tiff_out);
            if (n<0) fatal("failed to write TIFF image '"s+out_filename+"'");
        }
    }
    catch (usage_error& e) {
        emit_usage_error(e.what());
        return 2;
    }
    catch (fatal_error& e) {
        std::cerr << "blerp-test: " << e.what() << '\n';
        return 1;
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << '\n';
        return 1;
    }
}

