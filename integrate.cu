#include <iostream>
#include "datastructs.cpp"
#include "interp.cu"
#include <fstream>
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

__global__ void test(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, int MX, int MY, int MZ, int nT) {
	int parcel_id = blockIdx.x*blockDim.x + threadIdx.x;
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
        for (int tidx = 0; tidx < nT; ++tidx) {
            point[0] = parcels.xpos[parcel_id + parcels.nParcels * tidx];
            point[1] = parcels.ypos[parcel_id + parcels.nParcels * tidx];
            point[2] = parcels.zpos[parcel_id + parcels.nParcels * tidx];


            is_ugrd = true;
            is_vgrd = false;
            is_wgrd = false;
            pcl_u = interp3D(grid.xh, grid.yh, grid.zh, u_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = true;
            is_wgrd = false;
            pcl_v = interp3D(grid.xh, grid.yh, grid.zh, v_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = true;
            pcl_w = interp3D(grid.xh, grid.yh, grid.zh, w_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            // if the parcel has left the domain, exit
            if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                return;
            }
            else {

                // integrate X position forward by the U wind
                parcels.xpos[parcel_id + parcels.nParcels*(tidx + 1)] = point[0] + pcl_u * (1.0f/6.0f);
                // integrate Y position forward by the V wind
                parcels.ypos[parcel_id + parcels.nParcels*(tidx + 1)] = point[1] + pcl_v * (1.0f/6.0f);
                // integrate Z position forward by the W wind
                parcels.zpos[parcel_id + parcels.nParcels*(tidx + 1)] = point[2] + pcl_w * (1.0f/6.0f);
            }
        }
    }
}

/*This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff*/
void cudaIntegrateParcels(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, int MX, int MY, int MZ, int nT) {
    // pointers to device memory
    float *device_u_time_chunk, *device_v_time_chunk, *device_w_time_chunk;
    parcel_pos device_parcels;
    datagrid device_grid;

    // copy over our integer and long
    // constants to our device struct
    device_grid.X0 = grid.X0; device_grid.X1 = grid.X1;
    device_grid.Y0 = grid.Y0; device_grid.Y1 = grid.Y1;
    device_grid.Z0 = grid.Z0; device_grid.Z1 = grid.Z1;
    device_grid.NX = grid.NX; device_grid.NY = grid.NY;
    device_grid.NZ = grid.NZ; 
    device_parcels.nParcels = parcels.nParcels;

    // allocate device memory for our grid arrays
    //gpuErrchk( cudaMalloc(&(device_grid.xf), device_grid.NX*sizeof(float)) );
    //gpuErrchk( cudaMalloc(&(device_grid.yf), device_grid.NY*sizeof(float)) );
    //gpuErrchk( cudaMalloc(&(device_grid.zf), device_grid.NZ*sizeof(float)) );
    // we really don't need the staggered grid mesh sent
    gpuErrchk( cudaMalloc(&(device_grid.xh), device_grid.NX*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.yh), device_grid.NY*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.zh), device_grid.NZ*sizeof(float)) );
    // allocate the device memory for U/V/W
    gpuErrchk( cudaMalloc(&device_u_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_v_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_w_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    // allocate device memory for our parcel positions
    gpuErrchk( cudaMalloc(&(device_parcels.xpos), parcels.nParcels * (nT+1) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.ypos), parcels.nParcels * (nT+1) * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.zpos), parcels.nParcels * (nT+1) * sizeof(float)) );

    // copy the arrays to device memory
    gpuErrchk( cudaMemcpy(device_u_time_chunk, u_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_v_time_chunk, v_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_w_time_chunk, w_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_parcels.xpos, parcels.xpos, parcels.nParcels * (nT+1) * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.ypos, parcels.ypos, parcels.nParcels * (nT+1) * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.zpos, parcels.zpos, parcels.nParcels * (nT+1) * sizeof(float), cudaMemcpyHostToDevice) );

    // don't need the staggered mesh sent
    //gpuErrchk( cudaMemcpy(device_grid.xf, grid.xf, device_grid.NX*sizeof(float), cudaMemcpyHostToDevice) );
    //gpuErrchk( cudaMemcpy(device_grid.yf, grid.yf, device_grid.NY*sizeof(float), cudaMemcpyHostToDevice) );
    //gpuErrchk( cudaMemcpy(device_grid.zf, grid.zf, device_grid.NZ*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.xh, grid.xh, device_grid.NX*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.yh, grid.yh, device_grid.NY*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.zh, grid.zh, device_grid.NZ*sizeof(float), cudaMemcpyHostToDevice) );

    test<<<parcels.nParcels,1>>>(device_grid, device_parcels, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, MX, MY, MZ, nT);
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(parcels.xpos, device_parcels.xpos, parcels.nParcels * (nT+1) * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.ypos, device_parcels.ypos, parcels.nParcels * (nT+1) * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.zpos, device_parcels.zpos, parcels.nParcels * (nT+1) * sizeof(float), cudaMemcpyDeviceToHost) );


    cout << "FOLLOWING SINGLE PARCEL POST INTEGRATION" << endl;
    for (int t = 0; t < nT+1; ++t) {
        cout << "X: " << parcels.xpos[(int)(parcels.nParcels/2) + parcels.nParcels * t];
        cout << " Y: " << parcels.ypos[(int)(parcels.nParcels/2) + parcels.nParcels * t];
        cout << " Z: " << parcels.zpos[(int)(parcels.nParcels/2) + parcels.nParcels * t];
        cout << endl;
    }

    ofstream outfile;
    outfile.open("./result.csv");
    
    // loop over each parcel
    for (int pcl = 0; pcl < parcels.nParcels; ++ pcl) {
        // print the parcel start flag 
        outfile << "!Parcel " << pcl << endl; 
        // loop over the times
        for (int t = 0; t < nT+1; ++t) {
            // for each row: x position, y position, z position
            for (int row = 0; row < 3; ++row) {
                if (row == 0) outfile << parcels.xpos[pcl + t*parcels.nParcels] << ", ";
                if (row == 1) outfile << parcels.ypos[pcl + t*parcels.nParcels] << ", ";
                if (row == 2) outfile << parcels.zpos[pcl + t*parcels.nParcels] << endl;
            }
        }
        // parcel end flag
        outfile << "!End " << pcl << endl;
    }
}

#endif
