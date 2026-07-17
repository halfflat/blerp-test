# CPPFLAGS+=$(CRAY_ROCM_INCLUDE_OPTS)

HIPCC=hipcc

LDLIBS=-ltiff

blerp-test: runner.o kernels.o gpu_hip.o
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

runner.o: CXXFLAGS+=-std=c++20
runner.o: runner.cc kernels.h gpu.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

gpu_hip.o: gpu_hip.hip gpu.h
	$(HIPCC) $(CPPFLAGS) -c -o $@ $<

kernels.o: kernels.hip kernels.h
	$(HIPCC) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -f runner.o kernels.o gpu_hip.o
