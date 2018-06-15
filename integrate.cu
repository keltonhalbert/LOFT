#include <iostream>
#include "datastructs.cpp"
#ifndef INTEGRATE_CU
#define INTEGRATE_CU

using namespace std;
// this is an error checking helper function for processes
// that run on the GPU. Without calling this, the GPU can
// fail to execute but the program won't crash or report it.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      cout << cudaGetErrorString(code) << endl;
      if (abort) exit(code);
   }
}

// This is a GPU kernel for integrating a fluid parcel forward in time by increment dt.
// It expects the X, Y, and Z array pointers as well as the U, V, and W wind component pointers,
// which are all of length nParcels
__global__ void integrate(float *x_arr, float *y_arr, float *z_arr, float *u_arr, float *v_arr, float *w_arr, int nParcels, float dt) {
	// use the thread index to index the array
	int pidx = blockIdx.x*blockDim.x + threadIdx.x;

	// safety check to not access array memory out of bounds
	if (pidx < nParcels) {
		// integrate X position forward by the U wind
		x_arr[pidx] = x_arr[pidx] + u_arr[pidx] * dt;
		// integrate Y position forward by the V wind
		y_arr[pidx] = y_arr[pidx] + v_arr[pidx] * dt;
		// integrate Z position forward by the W wind
		z_arr[pidx] = z_arr[pidx] + w_arr[pidx] * dt;
	}
}

__global__ void test(parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, int MX, int MY, int MZ, int nT) {
    int parcel_id = blockIdx.x + threadIdx.x;
    if (parcel_id < parcels.nParcels) {
        printf("Parcel Number: %d\tParcel X: %f\tParcel Y: %f\t Parcel Z: %f\n", parcel_id, parcels.xpos[parcel_id], parcels.ypos[parcel_id], parcels.zpos[parcel_id]);
    }
}

/*This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff*/
void cudaIntegrateParcels(parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, int MX, int MY, int MZ, int nT) {
    // pointers to device memory
    float *device_u_time_chunk, *device_v_time_chunk, *device_w_time_chunk;
    parcel_pos device_parcels;
    device_parcels.nParcels = parcels.nParcels;

    // allocate the device memory
    gpuErrchk( cudaMalloc(&device_u_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_v_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_w_time_chunk, MX*MY*MZ*nT*sizeof(float)) );

    gpuErrchk( cudaMalloc(&(device_parcels.xpos), parcels.nParcels * nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.ypos), parcels.nParcels * nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.zpos), parcels.nParcels * nT*sizeof(float)) );

    // copy the arrays to device memory
    gpuErrchk( cudaMemcpy(device_u_time_chunk, u_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_v_time_chunk, v_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_w_time_chunk, w_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_parcels.xpos, parcels.xpos, parcels.nParcels * nT * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.ypos, parcels.ypos, parcels.nParcels * nT * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.zpos, parcels.zpos, parcels.nParcels * nT * sizeof(float), cudaMemcpyHostToDevice) );

    test<<<parcels.nParcels,1>>>(device_parcels, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, MX, MY, MZ, nT);
    gpuErrchk( cudaDeviceSynchronize() );



}

#endif
