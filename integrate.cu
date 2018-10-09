#include <iostream>
#include <stdio.h>
#include "datastructs.cpp"
#include "interp.cu"
#ifndef INTEGRATE_CU
#define INTEGRATE_CU

using namespace std;
#define P2(t,p,mt) (((p)*(mt))+(t))
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

__device__ void calc_zvort(datagrid grid, float *uarr, float *varr, float *zvort, int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xh[i+1] - grid.xh[i];
    float dy = grid.yh[j+1] - grid.yh[j];
    float dv = varr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float du = uarr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    zvort[arrayIndex(i, j, k, t, MX, MY, MZ)] = ( dv / dx ) - ( du / dy);
    //printf("%f, %i, %i, %i, %i\n", zvort[arrayIndex(i, j, k, t, MX, MY, MZ)], i, j, k, t);
}

/* Kernel for computing the components of vorticity
    and vorticity forcing terms. We do this using our domain subset containing the parcels
    instead of doing it locally for each parcel, as it would scale poorly for large 
    numbers of parcels. */
__global__ void calcvort(datagrid grid, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                        float *zvort, \
                        int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {

    // get our 3D index based on our blocks/threads
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];
    //printf("%i, %i, %i\n", i, j, k);

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX) && (j < MY) && (k < MZ)) { 
        if ((i+1 > MX) || (j+1) > MY) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            //calc_xvort(grid, u_time_chunk, v_time_chunk, w_time_chunk, xvort, MX, MY, MZ);
            //calc_yvort(grid, u_time_chunk, v_time_chunk, w_time_chunk, yvort, MX, MY, MZ);
            // calculate the Z component of vorticity
            //printf("%i, %i, %i, %i, %i, %i, %i\n", i, j, k, tidx, MX, MY, MZ);
            calc_zvort(grid, u_time_chunk, v_time_chunk, zvort, idx_4D, MX, MY, MZ);
        }
    }


}

__global__ void test(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                    int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {
	int parcel_id = blockIdx.x;
    // safety check to make sure our thread index doesn't
    // go out of our array bounds
    if (parcel_id < parcels.nParcels) {
        bool is_ugrd = false;
        bool is_vgrd = false;
        bool is_wgrd = false;

        float pcl_u, pcl_v, pcl_w;
        float point[3];

        // loop over the number of time steps we are
        // integrating over
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // GPU sanity test of data integrity
            point[0] = parcels.xpos[P2(tidx, parcel_id, totTime)];
            point[1] = parcels.ypos[P2(tidx, parcel_id, totTime)];
            point[2] = parcels.zpos[P2(tidx, parcel_id, totTime)];

            is_ugrd = true;
            is_vgrd = false;
            is_wgrd = false;
            pcl_u = interp3D(grid, u_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = true;
            is_wgrd = false;
            pcl_v = interp3D(grid, v_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = true;
            pcl_w = interp3D(grid, w_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);


            // integrate X position forward by the U wind
            point[0] += pcl_u * (1.0f/6.0f);
            // integrate Y position forward by the V wind
            point[1] += pcl_v * (1.0f/6.0f);
            // integrate Z position forward by the W wind
            point[2] += pcl_w * (1.0f/6.0f);
            if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                    point[0], point[1], point[2], grid.xh[0], grid.yh[0], grid.zh[0], grid.xh[grid.NX-1], grid.yh[grid.NY-1], grid.zh[grid.NZ-1]);
                return;
            }

            if (point[2] < grid.zf[0]) point[2] = grid.zf[0];


            parcels.xpos[P2(tidx+1, parcel_id, totTime)] = point[0]; 
            parcels.ypos[P2(tidx+1, parcel_id, totTime)] = point[1];
            parcels.zpos[P2(tidx+1, parcel_id, totTime)] = point[2];
            parcels.pclu[P2(tidx, parcel_id, totTime)] = pcl_u;
            parcels.pclv[P2(tidx, parcel_id, totTime)] = pcl_v;
            parcels.pclw[P2(tidx, parcel_id, totTime)] = pcl_w;
        }
    }
}

/*This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff*/
void cudaIntegrateParcels(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, int MX, int MY, int MZ, int nT, int totTime) {
    // pointers to device memory
    float *device_u_time_chunk, *device_v_time_chunk, *device_w_time_chunk;
    float *device_zvort_time_chunk;

    parcel_pos device_parcels;
    datagrid device_grid;

    int tStart, tEnd;
    tStart = 0;
    tEnd = nT;

    // copy over our integer and long
    // constants to our device struct
    device_grid.X0 = grid.X0; device_grid.X1 = grid.X1;
    device_grid.Y0 = grid.Y0; device_grid.Y1 = grid.Y1;
    device_grid.Z0 = grid.Z0; device_grid.Z1 = grid.Z1;
    device_grid.NX = grid.NX; device_grid.NY = grid.NY;
    device_grid.NZ = grid.NZ; 
    device_parcels.nParcels = parcels.nParcels;

    // allocate device memory for our grid arrays
    gpuErrchk( cudaMalloc(&(device_grid.xh), device_grid.NX*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.yh), device_grid.NY*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.zh), device_grid.NZ*sizeof(float)) );

    gpuErrchk( cudaMalloc(&(device_grid.xf), device_grid.NX*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.yf), device_grid.NY*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.zf), device_grid.NZ*sizeof(float)) );
    // allocate the device memory for U/V/W
    gpuErrchk( cudaMalloc(&device_u_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_v_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_w_time_chunk, MX*MY*MZ*nT*sizeof(float)) );

    //vorticity device arrays we have to calculate
    gpuErrchk( cudaMalloc(&device_zvort_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    // allocate device memory for our parcel positions
    gpuErrchk( cudaMalloc(&(device_parcels.xpos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.ypos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.zpos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclu), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclv), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclw), parcels.nParcels * totTime * sizeof(float)) );

    // copy the arrays to device memory
    gpuErrchk( cudaMemcpy(device_u_time_chunk, u_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_v_time_chunk, v_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_w_time_chunk, w_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_parcels.xpos, parcels.xpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.ypos, parcels.ypos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.zpos, parcels.zpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_grid.xh, grid.xh, device_grid.NX*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.yh, grid.yh, device_grid.NY*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.zh, grid.zh, device_grid.NZ*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_grid.xf, grid.xf, device_grid.NX*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.yf, grid.yf, device_grid.NY*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.zf, grid.zf, device_grid.NZ*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaDeviceSynchronize() );
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((MX/threadsPerBlock.x)+1, (MY/threadsPerBlock.y)+1, (MZ/threadsPerBlock.z)+1); 
    cout << "Calculating vorticity" << endl;
    cout << numBlocks.x << " " << numBlocks.y << " " << numBlocks.z << " " << endl;
    calcvort<<<numBlocks, threadsPerBlock>>>(device_grid, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, device_zvort_time_chunk, MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );
    cout << "End vorticity calc" << endl;
    test<<<parcels.nParcels,1>>>(device_grid, device_parcels, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(parcels.xpos, device_parcels.xpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.ypos, device_parcels.ypos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.zpos, device_parcels.zpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclu, device_parcels.pclu, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclv, device_parcels.pclv, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclw, device_parcels.pclw, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaDeviceSynchronize() );

    cudaFree(device_grid.xh);
    cudaFree(device_grid.yh);
    cudaFree(device_grid.zh);
    cudaFree(device_grid.xf);
    cudaFree(device_grid.yf);
    cudaFree(device_grid.zf);
    cudaFree(device_parcels.xpos);
    cudaFree(device_parcels.ypos);
    cudaFree(device_parcels.zpos);
    cudaFree(device_parcels.pclu);
    cudaFree(device_parcels.pclv);
    cudaFree(device_parcels.pclw);
    cudaFree(device_u_time_chunk);
    cudaFree(device_v_time_chunk);
    cudaFree(device_w_time_chunk);
    cudaFree(device_zvort_time_chunk);

    gpuErrchk( cudaDeviceSynchronize() );
    cout << "FINISHED CUDA" << endl;
}

#endif
