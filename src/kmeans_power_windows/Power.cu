
#include <stdio.h>
#include <stdint.h>

#include <device_functions.h>

// CUDA runtime
#include "cuda_runtime.h"
//#ifndef __CUDACC__  
//#define __CUDACC__
//#endif
#include "device_launch_parameters.h"
//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)
#include <assert.h>
// Helper functions and utilities to work with CUDA

//#include <helper_functions.h>
#include <helper_cuda.h>
#include <limits.h>
#include <time.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <Windows.h>
#include <limits.h>
#include <climits>
#include <nvml.h>
#include <math.h>
#include <conio.h>
#include <process.h>
#include <cuda_profiler_api.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cuda.h>

#include "kmeans_cuda_kernel.cu"
#include "Power.h"
//##########################################################################k

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM
#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE
//###########################################################################
#define DEVICE 0  /* may need to be changed */

#define NEAR_IDLE_DELTA 500  /* mW */
#define IDLE_DELTA 250  /* mW */
#define SAMPLE_DELAY 6 /* msec */
#define RAMP_DELAY 4 /* msec */
#define TIME_OUT 30    /* msec */
#define STABLE_COUNT 5  /* sec */

#define power2watts 0.001  /* mW -> W */
#define time2seconds 0.001  /* usec -> sec */
#define capacitance 840.0  /* msec */
#define ACTIVE_IDLE 55  /* W */
#define SAMPLES (1024*1024)  /* 4.3 hours */
//######################################################################k
extern "C"
DWORD WINAPI setup(void* data);									/* function prototype */
// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */
/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */



//float  **feature;				/* in: [npoints][nfeatures] */
//int      nfeatures;				/* number of attributes for each point */
//int      npoints;				/* number of data points */
//int      nclusters;				/* number of clusters */
//int     *membership;				/* which cluster the point belongs to */
//float  **clusters;  			/* coordinates of cluster centers */
//int     *new_centers_len;		/* number of elements in each cluster */
//float  **new_centers;				/* sum of elements in each cluster */

//######################################################################

static int p_sample[SAMPLES];  /* power */
unsigned __int64 t_sample[SAMPLES];  /* time */
static double truepower[SAMPLES];  /* true power */
static double max_power;  /* power cap in W */

unsigned __int64 endTimee;
unsigned __int64 startTimee;
unsigned __int64 freq;
double timerFrequency;

static int begin;
static int end;
const char *kernelName;


const int			  BLOCKDIM = 256;
const int        THREAD_PER_SM = 2048;
const int         MAX_BLOCKDIM = 1024;
const int MAX_SHARED_PER_BLOCK = 49152;
const int         CONSTANT_MEM = 49152;
const int          MEMORY_BANK = 32;
const int            WARP_SIZE = 32;
const int            SM = 15;


const int           NUM_SM = SM;
const int RESIDENT_THREADS = SM * THREAD_PER_SM;
const int   RESIDENT_WARPS = RESIDENT_THREADS / 32;

const int UNROLL_SIZE = 4096;
const int TOTAL_RECURSION = 128;
const int TEMPL_RECURSIONS = 1;

//------------------------------------------------------------------------------


template<typename T>
using ThroughputOP = void(*)(T&);


__device__ int       devTMP;

static_assert(UNROLL_SIZE % 2 == 0, "UNROLL_SIZE must be a multiple of 2");
static_assert(TOTAL_RECURSION % TEMPL_RECURSIONS == 0,
	"TOTAL_RECURSION must be a multiple of TEMPL_RECURSIONS_PARAM");
//###########################################################################################k
/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{
	num_blocks = npoints / num_threads;
	if (npoints % num_threads > 0)		/* defeat truncation */
		num_blocks++;

	num_blocks_perdim = sqrt((double)num_blocks);
	while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
		num_blocks_perdim++;

	num_blocks = num_blocks_perdim*num_blocks_perdim;

	/* allocate memory for memory_new[] and initialize to -1 (host) */
	membership_new = (int*)malloc(npoints * sizeof(int));
	for (int i = 0; i<npoints; i++) {
		membership_new[i] = -1;
	}

	/* allocate memory for block_new_centers[] (host) */
	block_new_centers = (float *)malloc(nclusters*nfeatures*sizeof(float));

	/* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
	cudaMalloc((void**)&feature_flipped_d, npoints*nfeatures*sizeof(float));
	cudaMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&feature_d, npoints*nfeatures*sizeof(float));

	/* invert the data array (kernel execution) */
	invert_mapping << <num_blocks, num_threads >> >(feature_flipped_d, feature_d, npoints, nfeatures);

	/* allocate memory for membership_d[] and clusters_d[][] (device) */
	cudaMalloc((void**)&membership_d, npoints*sizeof(int));
	cudaMalloc((void**)&clusters_d, nclusters*nfeatures*sizeof(float));


#ifdef BLOCK_DELTA_REDUCE
	// allocate array to hold the per block deltas on the gpu side

	cudaMalloc((void**)&block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
	//cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
	// allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
	cudaMalloc((void**)&block_clusters_d,
		num_blocks_perdim * num_blocks_perdim *
		nclusters * nfeatures * sizeof(float));
	//cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
#endif

}
/* -------------- allocateMemory() end ------------------- */


/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C"
void deallocateMemory()
{
	free(membership_new);
	free(block_new_centers);
	cudaFree(feature_d);
	cudaFree(feature_flipped_d);
	cudaFree(membership_d);

	cudaFree(clusters_d);
#ifdef BLOCK_CENTER_REDUCE
	cudaFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
	cudaFree(block_deltas_d);
#endif
}
/* -------------- deallocateMemory() end ------------------- */




//###########################################################################################
static nvmlDevice_t initAndTest()
{
	nvmlReturn_t result;
	nvmlDevice_t device;
	int power;

	result = nvmlInit();
	if (NVML_SUCCESS != result) {
		printf("failed to initialize NVML: %s\n", nvmlErrorString(result));
		exit(-1);
	}

	result = nvmlDeviceGetHandleByIndex(DEVICE, &device);
	if (NVML_SUCCESS != result) {
		printf("failed to get handle for device: %s\n", nvmlErrorString(result));
		exit(-1);
	}

	result = nvmlDeviceGetPowerUsage(device, (unsigned int *)&power);
	if (NVML_SUCCESS != result) {
		printf("failed to read power: %s\n", nvmlErrorString(result));
		exit(-1);
	}

	result = nvmlDeviceGetPowerManagementLimit(device, (unsigned int *)&power);
	if (NVML_SUCCESS != result) {
		printf("failed to read power limit: %s\n", nvmlErrorString(result));
		exit(-1);
	}
	max_power = power * power2watts;

	return device;
}


static void getSample(nvmlDevice_t device, int *power, unsigned __int64 *ltime, int samples)  /* mW usec */
{
	unsigned __int64  sampletime;
	nvmlReturn_t result;
	int samplepower;
	int sampless = samples;
	//Sleep(1);
	for (int i = 0; i < 2000; i++) {
		int a = i*i*i;
	}
	//while (getTime() < sampletime) {};
	result = nvmlDeviceGetPowerUsage(device, (unsigned int *)&samplepower);
	QueryPerformanceCounter((LARGE_INTEGER *)&sampletime);

	if (NVML_SUCCESS != result) {
		printf("failed to read power: %s\n", nvmlErrorString(result));
		return;
	}

	p_sample[samples] = samplepower;
	t_sample[samples] = sampletime;
	sampless++;

	if (samples >= SAMPLES) {
		printf("out of memory for storing samples\n");
		return;
	}

	if (samples >= 3) {
		int s = samples - 2;
		double tp = (p_sample[s] + capacitance * (p_sample[s + 1] - p_sample[s - 1]) / (t_sample[s + 1] - t_sample[s - 1])) * power2watts;
		if (tp < 0.0) tp = 0.0;
		if (tp > max_power) tp = max_power;
		truepower[s] = tp;
	}

	*power = samplepower;
	*ltime = sampletime;
}
/*
__device__ __forceinline__ unsigned int LaneID() {
	unsigned int ret;
	asm("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

template<typename T, ThroughputOP<T> OP_Function>
__global__ void MeasureKernel() {
	T R1 = threadIdx.x;


	OP_Function(R1);

	if (LaneID() > 0)
		return;

}

template<int RECURSIONS>
__device__ __forceinline__
void Integer_ADD_Support(int& R1) {
#pragma unroll
	for (int i = 0; i < UNROLL_SIZE; i++) {
		asm volatile("add.s32 %0, %1, %2;"
			: "=r"(R1) : "r"(R1), "r"(R1) : "memory");
	}
	Integer_ADD_Support<RECURSIONS - 1>(R1);
}
template<>
__device__ __forceinline__
void Integer_ADD_Support<0>(int& R1) {}

__device__ __forceinline__
void Integer_ADD(int& R1) {
	for (int i = 0; i < TOTAL_RECURSION / TEMPL_RECURSIONS; i++)
		Integer_ADD_Support<TEMPL_RECURSIONS>(R1);
}
*/

void intAdd(float  **feature,				/* in: [npoints][nfeatures] */
	int      nfeatures,				/* number of attributes for each point */
	int      npoints,				/* number of data points */
	int      nclusters,				/* number of clusters */
	int     *membership,				/* which cluster the point belongs to */
	float  **clusters,				/* coordinates of cluster centers */
	int     *new_centers_len,		/* number of elements in each cluster */
	float  **new_centers,			/* sum of elements in each cluster */
	dim3	threads,
	dim3	grid)
{
	const int GRIDDIM = RESIDENT_THREADS / BLOCKDIM;
	cudaError_t cudaStatus;
	//printf("Resident=%d\n", RESIDENT_THREADS);
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	begin = 1;
	QueryPerformanceCounter((LARGE_INTEGER *)&startTimee);
	printf("launching........\n");
	//////////////////////////////////////////////////////////////////////////
	//MeasureKernel<int, Integer_ADD> <<<GRIDDIM, BLOCKDIM >> >();
	kmeansPoint << < grid, threads >> >(feature_d,
		nfeatures,
		npoints,
		nclusters,
		membership_d,
		clusters_d,
		block_clusters_d,
		block_deltas_d);
	//////////////////////////////////////

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	goto Error;
	}
	cudaDeviceSynchronize();
	int iter = 10000000;
	begin = 1;
	QueryPerformanceCounter((LARGE_INTEGER *)&startTimee);
	for (int i = 0; i < iter; i++)
	{
		//MeasureKernel<int, Integer_ADD> << <GRIDDIM, BLOCKDIM >> >();
	}
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaDeviceSynchronize();
	end = 1;
	QueryPerformanceCounter((LARGE_INTEGER *)&endTimee);
	//printf("end=%lld", endTimee);
	printf("timeeeeeee=%f \n", ((endTimee - startTimee))*timerFrequency);

Error:

	end = 1;

	return;
}
//=====================================================================main
int main(int argc, char** argv) {
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	timerFrequency = (1.0 / freq);

	unsigned __int64 now;

	t_sample[0] = 0;
	FILE *f;
	char filename[1100];
	int samples = 0;
	int i, count, active_samples;
	nvmlDevice_t device;
	int power, prevpower, nearidlepower, diff;
	unsigned __int64 l_time, timeout, endtime;
	double activetime, activeenergy, mindt;
	LoadLibrary("C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll");
	printf("K20Power 1.0\n");

	//char * my_kernel = "test";
	char * my_kernel = "int32add";
	sprintf(filename, "../../results/K20Power_%s.trace", my_kernel);
	f = fopen(filename, "wt");
	fprintf(f, "K20Power 1.0\t#version\n");

	fprintf(f, "\t#command line\n\n");

	device = initAndTest();

	samples++;
	getSample(device, &power, &l_time, samples);
	QueryPerformanceCounter((LARGE_INTEGER *)&now);
	printf("power=%d\t", power);
	timeout = now*timerFrequency + TIME_OUT;
	count = 0;
	do {
		prevpower = power;
		for (int i = 0; i < 2000; i++)
			int a = i*i*i*i;
		getSample(device, &power, &l_time, samples);
		samples++;
		printf("power=%d\t", power);
		count++;
		diff = power - prevpower;
		if (diff < 0) diff = -diff;
		if (diff >= IDLE_DELTA) count = 0;
	} while ((count < STABLE_COUNT) && (l_time*timerFrequency < timeout));

	if (l_time*timerFrequency >= timeout) {
		printf("timed out waiting for idle power to stabilize\n");
		exit(-1);
	}

	getSample(device, &power, &l_time, samples);
	samples++;
	printf("Power=%d\t", power);
	QueryPerformanceCounter((LARGE_INTEGER *)&now);
	endtime = l_time*timerFrequency + RAMP_DELAY;
	do {
		QueryPerformanceCounter((LARGE_INTEGER *)&now);
	} while (now*timerFrequency < endtime);
	getSample(device, &power, &l_time, samples);
	samples++;
	printf("Power=%d\t", power);
	nearidlepower = power + NEAR_IDLE_DELTA;
	printf("nearIDlePower=%d\n", nearidlepower);


	end = 0;



	samples++;
	printf("Power=%d\n", power);
	//****************************************************************************start thread
		// as done in the CUDA start/help document provided
		
	HANDLE thread = CreateThread(NULL, 0, setup, NULL, 0, NULL);
	//_beginthread(setup, 0, NULL);
	QueryPerformanceCounter((LARGE_INTEGER *)&now);
	getSample(device, &power, &l_time, samples);

	timeout = now*timerFrequency + TIME_OUT;
	while (l_time*timerFrequency < timeout && end == 0) {
		getSample(device, &power, &l_time, samples);
		samples++;

		if (end == 0) {
			QueryPerformanceCounter((LARGE_INTEGER *)&now);
			timeout = now*timerFrequency + TIME_OUT;
		}

	}
	fprintf(f, "end of running kernel=%.6f\n", (endTimee - t_sample[1])*timerFrequency);
	fprintf(f, "start of running kernel=%.6f\n", (startTimee - t_sample[1])*timerFrequency);
	fprintf(f, "elapsed Time=%.6f\n", (endTimee - startTimee)*timerFrequency);


	printf("here must be the end of Kernel\n");
	getSample(device, &power, &l_time, samples);
	samples++;
	printf("Power=%d\n", power);
	getSample(device, &power, &l_time, samples);
	samples++;
	printf("Power=%d\n", power);
	samples--;
	active_samples = 0;
	activetime = 0.0;
	activeenergy = 0.0;

	mindt = TIME_OUT;
	for (i = 1; i < samples; i++) {
		if (truepower[i] > ACTIVE_IDLE) {
			active_samples++;
			double dt = (t_sample[i] - t_sample[i - 1]) * timerFrequency;
			if (mindt > dt) mindt = dt;
			activetime += dt;
			activeenergy += dt * truepower[i];
		}
	}

	fprintf(f, "%.4f\t#active time [s]\n", activetime);
	fprintf(f, "%.4f\t#active energy [J]\n", activeenergy);

	fprintf(f, "\ntime [s]\tpower [W]\ttrue power [W]\n");
	for (i = 1; i < samples; i++) {
		fprintf(f, "%.6f \t %.6f \t %.6f\n", (t_sample[i] - t_sample[1]) * timerFrequency, p_sample[i] * power2watts, truepower[i]);
	}
	fclose(f);
	printf("samples=%d\n", samples);
	nvmlShutdown();
	getch();
	return 0;
}
//#########################################################################k
/* ------------------- kmeansCuda() ------------------------ */
//extern "C"
int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
	int      nfeatures,				/* number of attributes for each point */
	int      npoints,				/* number of data points */
	int      nclusters,				/* number of clusters */
	int     *membership,				/* which cluster the point belongs to */
	float  **clusters,				/* coordinates of cluster centers */
	int     *new_centers_len,		/* number of elements in each cluster */
	float  **new_centers				/* sum of elements in each cluster */
	)
{
	int delta = 0;			/* if point has moved */
	int i, j;				/* counters */


	cudaSetDevice(1);

	/* copy membership (host to device) */
	cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice);

	/* copy clusters (host to device) */
	cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);

	/* set up texture */
	cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
	t_features.filterMode = cudaFilterModePoint;
	t_features.normalized = false;
	t_features.channelDesc = chDesc0;

	if (cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
		printf("Couldn't bind features array to texture!\n");

	cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
	t_features_flipped.filterMode = cudaFilterModePoint;
	t_features_flipped.normalized = false;
	t_features_flipped.channelDesc = chDesc1;

	if (cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
		printf("Couldn't bind features_flipped array to texture!\n");

	cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
	t_clusters.filterMode = cudaFilterModePoint;
	t_clusters.normalized = false;
	t_clusters.channelDesc = chDesc2;

	if (cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float)) != CUDA_SUCCESS)
		printf("Couldn't bind clusters array to texture!\n");

	/* copy clusters to constant memory */
	cudaMemcpyToSymbol("c_clusters", clusters[0], nclusters*nfeatures*sizeof(float), 0, cudaMemcpyHostToDevice);


	/* setup execution parameters.
	changed to 2d (source code on NVIDIA CUDA Programming Guide) */
	dim3  grid(num_blocks_perdim, num_blocks_perdim);
	dim3  threads(num_threads_perdim*num_threads_perdim);

	/* execute the kernel */
	//int *a = 0;
	intAdd(feature,
		nfeatures,
		npoints,
		nclusters,
		membership,
		clusters,
		new_centers_len,
		new_centers, threads, grid);

	cudaThreadSynchronize();

	/* copy back membership (device to host) */
	cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost);

#ifdef BLOCK_CENTER_REDUCE
	/*** Copy back arrays of per block sums ***/
	float * block_clusters_h = (float *)malloc(
		num_blocks_perdim * num_blocks_perdim *
		nclusters * nfeatures * sizeof(float));

	cudaMemcpy(block_clusters_h, block_clusters_d,
		num_blocks_perdim * num_blocks_perdim *
		nclusters * nfeatures * sizeof(float),
		cudaMemcpyDeviceToHost);
#endif
#ifdef BLOCK_DELTA_REDUCE
	int * block_deltas_h = (int *)malloc(
		num_blocks_perdim * num_blocks_perdim * sizeof(int));

	cudaMemcpy(block_deltas_h, block_deltas_d,
		num_blocks_perdim * num_blocks_perdim * sizeof(int),
		cudaMemcpyDeviceToHost);
#endif

	/* for each point, sum data points in each cluster
	and see if membership has changed:
	if so, increase delta and change old membership, and update new_centers;
	otherwise, update new_centers */
	delta = 0;
	for (i = 0; i < npoints; i++)
	{
		int cluster_id = membership_new[i];
		new_centers_len[cluster_id]++;
		if (membership_new[i] != membership[i])
		{
#ifdef CPU_DELTA_REDUCE
			delta++;
#endif
			membership[i] = membership_new[i];
		}
#ifdef CPU_CENTER_REDUCE
		for (j = 0; j < nfeatures; j++)
		{
			new_centers[cluster_id][j] += feature[i][j];
		}
#endif
	}


#ifdef BLOCK_DELTA_REDUCE	
	/*** calculate global sums from per block sums for delta and the new centers ***/

	//debug
	//printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
	for (i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		//printf("block %d delta is %d \n",i,block_deltas_h[i]);
		delta += block_deltas_h[i];
	}

#endif
#ifdef BLOCK_CENTER_REDUCE	

	for (int j = 0; j < nclusters; j++) {
		for (int k = 0; k < nfeatures; k++) {
			block_new_centers[j*nfeatures + k] = 0.f;
		}
	}

	for (i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
		for (int j = 0; j < nclusters; j++) {
			for (int k = 0; k < nfeatures; k++) {
				block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
			}
		}
	}


#ifdef CPU_CENTER_REDUCE
	//debug
	/*for(int j = 0; j < nclusters;j++) {
	for(int k = 0; k < nfeatures;k++) {
	if(new_centers[j][k] >	1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 * block_new_centers[j*nfeatures + k]) {
	printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
	}
	}
	}*/
#endif

#ifdef BLOCK_CENTER_REDUCE
	for (int j = 0; j < nclusters; j++) {
		for (int k = 0; k < nfeatures; k++)
			new_centers[j][k] = block_new_centers[j*nfeatures + k];
	}
#endif

#endif

	return delta;

}
/* ------------------- kmeansCuda() end ------------------------ */
//#########################################################################
