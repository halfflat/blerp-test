# CPPFLAGS+=$(CRAY_ROCM_INCLUDE_OPTS)

HIPCC=hipcc

LDLIBS=-ltiff

blerp-test: runner.o kernels.o gpu_hip.o
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

runner.o: override CXXFLAGS+=-std=c++20
runner.o: runner.cc kernels.h gpu.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

gpu_hip.o: gpu_hip.hip gpu_hip.h gpu.h
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) -c -o $@ $<

kernels.o: kernels.hip kernels.h gpu_hip.hip gpu_hip.h gpu.h
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -f runner.o kernels.o gpu_hip.o

realclean: clean
	rm -f blerp-test
