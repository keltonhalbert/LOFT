#include <iostream>
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


__global__ void test(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                    float *uparcels, float *vparcels, float *wparcels, int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {
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
        int tLocal = 0;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // GPU sanity test of data integrity
            point[0] = parcels.xpos[P2(tidx, parcel_id, totTime)];
            point[1] = parcels.ypos[P2(tidx, parcel_id, totTime)];
            point[2] = parcels.zpos[P2(tidx, parcel_id, totTime)];


            is_ugrd = true;
            is_vgrd = false;
            is_wgrd = false;
            pcl_u = interp3D(grid.xh, grid.yh, grid.zh, u_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tLocal, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = true;
            is_wgrd = false;
            pcl_v = interp3D(grid.xh, grid.yh, grid.zh, v_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tLocal, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = true;
            pcl_w = interp3D(grid.xh, grid.yh, grid.zh, w_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tLocal, MX, MY, MZ);

            uparcels[P2(tidx, parcel_id, totTime)] = pcl_u;
            vparcels[P2(tidx, parcel_id, totTime)] = pcl_v;
            wparcels[P2(tidx, parcel_id, totTime)] = pcl_w;

            // if the parcel has left the domain, exit
            tLocal += 1;
            if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                        point[0], point[1], point[2], grid.xh[0], grid.yh[0], grid.zh[0], grid.xh[grid.NX-1], grid.yh[grid.NY-1], grid.zh[grid.NZ-1]);
                return;
            }
            else {
                // integrate X position forward by the U wind
                point[0] += pcl_u * (1.0f/6.0f);
                // integrate Y position forward by the V wind
                point[1] += pcl_v * (1.0f/6.0f);
                // integrate Z position forward by the W wind
                point[2] += pcl_w * (1.0f/6.0f);

                parcels.xpos[P2(tidx+1, parcel_id, totTime)] = point[0]; 
                parcels.ypos[P2(tidx+1, parcel_id, totTime)] = point[1];
                parcels.zpos[P2(tidx+1, parcel_id, totTime)] = point[2];

            }
        }
    }
}

/*This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff*/
void cudaIntegrateParcels(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                            float *uparcels, float *vparcels, float *wparcels, int MX, int MY, int MZ, int nT, int totTime) {
    // pointers to device memory
    float *device_u_time_chunk, *device_v_time_chunk, *device_w_time_chunk;
    float *device_uparcels, *device_vparcels, *device_wparcels;

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
    // allocate the device memory for U/V/W
    gpuErrchk( cudaMalloc(&device_u_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_v_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_w_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    // allocate device memory for our parcel positions
    gpuErrchk( cudaMalloc(&(device_parcels.xpos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.ypos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.zpos), parcels.nParcels * totTime * sizeof(float)) );
    // allocate memory for tracking u/v/w along parcel
    gpuErrchk( cudaMalloc(&(device_uparcels), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_vparcels), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_wparcels), parcels.nParcels * totTime * sizeof(float)) );

    // copy the arrays to device memory
    gpuErrchk( cudaMemcpy(device_u_time_chunk, u_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_v_time_chunk, v_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_w_time_chunk, w_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_parcels.xpos, parcels.xpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.ypos, parcels.ypos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.zpos, parcels.zpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );

    // don't need the staggered mesh sent
    gpuErrchk( cudaMemcpy(device_grid.xh, grid.xh, device_grid.NX*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.yh, grid.yh, device_grid.NY*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.zh, grid.zh, device_grid.NZ*sizeof(float), cudaMemcpyHostToDevice) );
    test<<<parcels.nParcels,1>>>(device_grid, device_parcels, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, \
                                 device_uparcels, device_vparcels, device_wparcels, MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(parcels.xpos, device_parcels.xpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.ypos, device_parcels.ypos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.zpos, device_parcels.zpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(uparcels, device_uparcels, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(vparcels, device_vparcels, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(wparcels, device_wparcels, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaFree(device_grid.xh);
    cudaFree(device_grid.yh);
    cudaFree(device_grid.zh);
    cudaFree(device_parcels.xpos);
    cudaFree(device_parcels.ypos);
    cudaFree(device_parcels.zpos);
    cudaFree(device_uparcels);
    cudaFree(device_vparcels);
    cudaFree(device_wparcels);

    cout << "FINISHED CUDA" << endl;
}

#endif
