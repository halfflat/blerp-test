#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <tiffio.h>

#include <kernels.h>

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

void emit_usage_error(const std::string& msg) {
    if (!msg.empty()) std::cerr << "blerp-test: " << msg << '\n';
    std::cerr << "usage: blerp-test METHOD SAMPLING FILE [OUTFILE]\n\n    METHOD\t\tOne of array, texture, precomp.\n    SAMPLING\t\tDegree of oversampling in X and Y axes.\n    FILE\t\tTIFF file (greyscale) for sampling.\n"
        "    OUTFILE\t\t(Optional) write oversampled image to OUTFILE.\n\n";
    return 1;
}

void hip_check(hipError_t err, std::source_location where = std::source_location::current()) {
    if (!err) return;
    fatal(std::format("HIP error line {}: {}", where.line, hipGetErrorString(err)));
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
        else return usage();

        char* end = nullptr;
        double over = std::strtod(argv[2], &end);
        if (!end || *end) return usage();

        const char* src_filename = argv[3];
        TIFF* tiff_src = TIFFOpen(src_filename, "r");
        if (!tiff_src) fatal(std::format("failed to open TIFF image '{}'", src_filename));

        uint32_t src_h = 0, src_w = 0;
        TIFFGetField(tiff_src, TIFFTAG_IMAGEWIDTH, &src_w);
        TIFFGetField(tiff_src, TIFFTAG_IMAGELENGTH, &src_h);
        if (src_h==0 || src_w==0) return fatal("Failed to get image dimensions");

        uint16_t src_channels = 0, src_bpp = 0, src_fmt = 0;
        TIFFGetField(tiff_src, TIFFTAG_SAMPLESPERPIXEL, &src_channels);
        TIFFGetField(tiff_src, TIFFTAG_BITSPERSAMPLE, &src_bpp);
        TIFFGetFieldDefaulted(tiff_src, TIFFTAG_SAMPLEFORMAT, &src_fmt, uint16_t(1));
        if (src_channels!=1) return fatal("Expected greyscale (single-channel) image");

        if (src_fmt!=SAMPLEFORMAT_UINT && src_fmt!=SAMPLEFORMAT_IEEEFP) return fatal("unsupported TIFF sample format");

        uint16_t src_photometric = 0, src_orientation = 0;
        TIFFGetField(tiff_src, TIFFTAG_PHOTOMETRIC, &src_photometric);
        TIFFGetFieldDefaulted(tiff_src, TIFFTAG_ORIENTATION, &src_orientation, uint16_t(1));

        std::vector<float> src_data(src_h*src_w);
        float pxl_scale = float(1ull<<(src_bpp-1));

        auto scanline_size = TIFFScanlineSize(tiff_src);
        unsigned bytes_per_pixel = scanline_size/src_w;
        if ((bytes_per_pixel&(bytes_per_pixel-1))!=0 || bytes_per_pixel>4) return fatal("can't work with this number of bytes per pixel in scanline");

        std::vector<char> scanline_raw(scanline_size);

        for (uint32_t row = 0; row<src_h; ++row) {
            if (TIFFReadScanline(tiff_src, &scanline_raw[0], row, 0)<0) return fatal("TIFFReadScaline failed");

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
                return fatal("internal logic error");
            }
        }

        uint32_t out_w = uint32_t(std::round(over*src_w));
        uint32_t out_h = uint32_t(std::round(over*src_h));
        std::vector<float> out_data(out_h*out_w);

        void* src_dev_ptr = nullptr;
        void* out_dev_ptr = nullptr;
        void* pre_dev_ptr = nullptr;

        hip_assert(hipInit(0));
        hip_assert(hipMalloc(&src_dev_ptr, src_data.size()*sizeof(float)));
        hip_assert(hipMalloc(&out_dev_ptr, out_data.size()*sizeof(float)));
        hip_assert(hipMalloc(&pxd_dev_ptr, src_w*src_h*sizeof(pixel_data)));
        hip_assert(hipMemcpy(src_dev_ptr, &src_data[0], src_data.size()*sizeof(float), hipMemcpyHostToDevice));

        hipEvent_t ev_setup, ev_start, ev_stop;

        hip_assert(hipEventCreate(&ev_setup));
        hip_assert(hipEventCreate(&ev_start));
        hip_assert(hipEventCreate(&ev_stop));

        const unsigned n_iter = 1;

        hipEventRecord(ev_setup, 0);
        switch (method) {
        case method_array:
            hipEventRecord(ev_start, 0);
            for (unsigned i = 0; i<n_iter; ++i) run_subsample_image_array(src_dev_ptr, src_w, src_h, out_dev_ptr, ouw_w, out_h);
            break;
        case method_precomp:
            run_make_pixel_delta_array(src_dev_ptr, src_w, src_h, pxd_dev_ptr);
            hipEventRecord(ev_start, 0);
            for (unsigned i = 0; i<n_iter; ++i) run_subsample_pixel_delta_array (pxd_dev_ptr, src_w, src_h, out_dev_ptr, ouw_w, out_h);
            break;
        case method_texture:
            fatal("unimplemented");
        }
        hipEventRecord(ev_stop, 0);
        hip_assert(hipEventSynchronize(ev_stop);

        float time_setup = 0, time_run = 0;
        hipEventElapsedTime(&time_setup, ev_setup, ev_start);
        hipEventElapsedTime(&time_run, ev_start, ev_stop);

        std::cout << "normalized kernel execution times (ms):\nsetup: " << time_setup << "\nsampling: " << time_run/n_iter << '\n';

        hip_assert(hipMemcpy(out_dev_ptr, &out_data[0], src_data.size()*sizeof(float), hipMemcpyDeviceToHost));
        if (out_filename) {
            TIFF* tiff_out = TIFFOpen(out_filename, "w");
            if (!tiff_out) return fatal("failed to open '"s+out_filename+"' for writing");

            TIFFSetField(tiff_out, TIFFTAG_IMAGEWIDTH, out_w);
            TIFFSetField(tiff_out, TIFFTAG_IMAGELENGTH, out_h);
            TIFFSetField(tiff_out, TIFFTAG_BITSPERSAMPLE, 8);
            TIFFSetField(tiff_out, TIFFTAG_SAMPLESPERPIXEL, 1);
            TIFFSetField(tiff_out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            TIFFSetField(tiff_out, TIFFTAG_PHOTOMETRIC, src_photometric);
            TIFFSetField(tiff_out, TIFFTAG_ORIENTATION, src_orientation);
            TIFFSetField(tiff_out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(tiff_out, TIFFTAG_ROWSPERSTRIP, out_h);

            std::vector<uint16_t> out_data_u16(out_h*out_w);
            std::transform(out_data, out_data+out_h*out_w, &out_data_u16[0], [](float p) { return uint8_t(255*std::clamp(p, 0.f, 1.f)); });

            auto n = TIFFWriteEncodedStrip(tiff_out, 0, &out_data_u16[0], out_w*out_h*sizeof(uint16_t));
            TIFFClose(tiff_out);
            if (n<0) return fatal("failed to write TIFF image '"s+out_filename+"'");
        }
    }
    catch (usage_error& e) {
        usage(e.what());
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

