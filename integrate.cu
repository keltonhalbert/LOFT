#include <iostream>
#include <stdio.h>
#include "datastructs.cu"
#include "macros.cpp"
#include "interp.cu"
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



__device__ void calc_xvort(datagrid *grid, float *wstag, float *vstag, float *xvort, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
}

__device__ void calc_yvort(datagrid *grid, float *ustag, float *wstag, float *yvort, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

}

__device__ void calc_zvort(datagrid *grid, float *ustag, float *vstag, float *zvort, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

}

/* When doing the parcel trajectory integration, George Bryan does
   some fun stuff with the lower boundaries of the arrays, presumably
   to prevent the parcels from exiting out the bottom of the domain
   or experience artificial values */
__global__ void applyMomentumBC(float *ustag, float *vstag, float *wstag, int NX, int NY, int NZ, int tStart, int tEnd) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    // this is done for easy comparison to CM1 code
    int ni = NX; int nj = NY; int nk = NZ;

    // this is a lower boundary condition, so only when k is 0
    // also this is on the u staggered mesh
    if (( j >= 0 ) && ( i >= 0) && ( j < nj+1) && ( i < ni+2) && ( k == 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // use the u stagger macro to handle the
            // proper indexing
            UA4D(i, j, 0, tidx) = UA4D(i, j, 1, tidx);
        }
    }
    
    // do the same but now on the v staggered grid
    if (( j >= 0 ) && ( i >= 0) && ( j < nj+2) && ( i < ni+1) && ( k == 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // use the v stagger macro to handle the
            // proper indexing
            VA4D(i, j, 0, tidx) = VA4D(i, j, 1, tidx);
        }
    }

    // do the same but now on the w staggered grid
    if (( j >= 0 ) && ( i >= 0) && ( j < nj+1) && ( i < ni+1) && ( k == 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // use the w stagger macro to handle the
            // proper indexing
            WA4D(i, j, 0, tidx) = -1*WA4D(i, j, 2, tidx);
        }
    }
}

/* Kernel for computing the components of vorticity
    and vorticity forcing terms. We do this using our domain subset containing the parcels
    instead of doing it locally for each parcel, as it would scale poorly for large 
    numbers of parcels. */
__global__ void calcvort(datagrid *grid, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                        float *xvort, float *yvort, float *zvort, \
                        int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {

    // get our 3D index based on our blocks/threads
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];
    //printf("%i, %i, %i\n", i, j, k);

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort(grid, w_time_chunk, v_time_chunk, xvort, idx_4D, MX, MY, MZ);
            calc_yvort(grid, u_time_chunk, w_time_chunk, yvort, idx_4D, MX, MY, MZ);
            calc_zvort(grid, u_time_chunk, v_time_chunk, zvort, idx_4D, MX, MY, MZ);
        }
    }
}

__global__ void integrate(datagrid *grid, parcel_pos *parcels, integration_data *data, \
                          int tStart, int tEnd, int totTime, int direct) {

	int parcel_id = blockIdx.x;
    // safety check to make sure our thread index doesn't
    // go out of our array bounds
    if (parcel_id < parcels->nParcels) {
        bool is_ugrd = false;
        bool is_vgrd = false;
        bool is_wgrd = false;

        float pcl_u, pcl_v, pcl_w;
        float point[3];

        // loop over the number of time steps we are
        // integrating over
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // GPU sanity test of data integrity
            point[0] = parcels->xpos[PCL(tidx, parcel_id, totTime)];
            point[1] = parcels->ypos[PCL(tidx, parcel_id, totTime)];
            point[2] = parcels->zpos[PCL(tidx, parcel_id, totTime)];

            is_ugrd = true;
            is_vgrd = false;
            is_wgrd = false;
            pcl_u = interp3D(grid, data->u_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            is_ugrd = false;
            is_vgrd = true;
            is_wgrd = false;
            pcl_v = interp3D(grid, data->v_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = true;
            pcl_w = interp3D(grid, data->w_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            
            // integrate X position forward by the U wind
            point[0] += pcl_u * (1.0f/6.0f) * direct;
            // integrate Y position forward by the V wind
            point[1] += pcl_v * (1.0f/6.0f) * direct;
            // integrate Z position forward by the W wind
            point[2] += pcl_w * (1.0f/6.0f) * direct;
            if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                    point[0], point[1], point[2], grid->xh[0], grid->yh[0], grid->zh[0], grid->xh[grid->NX-1], grid->yh[grid->NY-1], grid->zh[grid->NZ-1]);
                return;
            }


            parcels->xpos[PCL(tidx+1, parcel_id, totTime)] = point[0]; 
            parcels->ypos[PCL(tidx+1, parcel_id, totTime)] = point[1];
            parcels->zpos[PCL(tidx+1, parcel_id, totTime)] = point[2];
        }
    }
}

/*This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff*/
void cudaIntegrateParcels(datagrid *grid, integration_data *data, parcel_pos *parcels, int nT, int totTime, int direct) {

    int tStart, tEnd;
    tStart = 0;
    tEnd = nT;
    int NX, NY, NZ;
    NX = grid->NX;
    NY = grid->NY;
    NZ = grid->NZ;


    gpuErrchk( cudaDeviceSynchronize() );
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((NX/threadsPerBlock.x)+1, (NY/threadsPerBlock.y)+1, (NZ/threadsPerBlock.z)+1); 

    for (int t = 0; t < nT; ++t) {
        for (int j = 0; j < grid->NY; ++j) {
            for (int i = 0; i < grid->NX; ++i) {
                float *wstag = data->w_4d_chunk;
                float *vstag = data->v_4d_chunk;
                float *ustag = data->u_4d_chunk;
                float *buf0 = data->pres_4d_chunk;

                cout << "Time: " << t;
                cout << " W momentum pre: " << (WA4D(i, j, 0, t) == -1*WA4D(i, j, 2, t));
                cout << " V momentum pre: " << (VA4D(i, j, 0, t) == VA4D(i, j, 1, t));
                cout << " U momentum pre: " << (UA4D(i, j, 0, t) == UA4D(i, j, 1, t)) << endl;
                if (t == nT - 1) {
                    //for (int k = 0; k < grid->NZ; ++k) {
                    //    cout << VA4D(i, j, k, t) << " ";
                    //}
                    //cout << endl;
                }
            }
        }
    }
    cout << endl;
    applyMomentumBC<<<numBlocks, threadsPerBlock>>>(data->u_4d_chunk, data->v_4d_chunk, data->w_4d_chunk, NX, NY, NZ, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    for (int t = 0; t < nT; ++t) {
        for (int j = 0; j < grid->NY; ++j) {
            for (int i = 0; i < grid->NX; ++i) {
                float *wstag = data->w_4d_chunk;
                float *vstag = data->v_4d_chunk;
                float *ustag = data->u_4d_chunk;
                float *buf0 = data->pres_4d_chunk;

                cout << "Time: " << t;
                cout << " W momentum: " << (WA4D(i, j, 0, t) == -1*WA4D(i, j, 2, t));
                cout << " V momentum: " << (VA4D(i, j, 0, t) == VA4D(i, j, 1, t));
                cout << " U momentum: " << (UA4D(i, j, 0, t) == UA4D(i, j, 1, t)) << endl;
                if (t == nT - 1) {
                    //for (int k = 0; k < grid->NZ; ++k) {
                    //    cout << VA4D(i, j, k, t) << " ";
                    //}
                    //cout << endl;
                }
            }
        }
    }
    cout << endl;
    //integrate(grid, parcels, data, nT, totTime, direct);
}

#endif
