nvcc_options= -O3 -arch=sm_35 -lnvidia-ml

monitor: power.cu nvml.h
	nvcc $(nvcc_options) power.cu -o monitor

clean:
	rm monitor
