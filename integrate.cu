#include <iostream>
#include <stdio.h>
#include "datastructs.cu"
#include "macros.cpp"
#include "interp.cu"
#include "turb.cu"
#include "vort.cu"
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


/*  Execute all of the required kernels on the GPU that are necessary for computing the 3
    components of vorticity. The idea here is that we're building wrappers on wrappers to
    simplify the process for the end user that just wants to calculate vorticity. This is
    also a necessary adjustment because the tendency calculations will require multiple
    steps, so transitioning this block of code as a proof of concept for how the programming
    model should work. */
void doCalcVort(datagrid *grid, integration_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock) {
    // calculate the three compionents of vorticity
    calcvort<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // apply the lower boundary condition to the horizontal
    // components of vorticity
    applyVortBC<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // Average the vorticity to the scalar grid using the temporary
    // arrays we allocated. After doing the averaging, we have to 
    // set the pointers to the temporary arrays as the new xvort,
    // yvort, and zvort, and set the old x/y/zvort arrays as the new
    // temporary arrays. Note: may have to zero those out in the future...
    doVortAvg<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

    zeroTemArrays<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
} 

void doCalcVortTend(datagrid *grid, integration_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock) {

    // Compute the vorticity tendency due to stretching. These conveniently
    // end up on the scalar grid, and no extra steps are required. This will
    // compute the tendency for all 3 components of vorticity. 
    calcvortstretch<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    applyVortTendBC<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // Compute the vertical vorticity tendency due to tilting. We have to do 
    // each component individually because we have to average the arrays back
    // to the scalar grid. It's a mess. 
    calcxvorttilt<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    applyVortTendBC<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    doXVortTiltAvg<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    zeroTemArrays<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

    calcyvorttilt<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    applyVortTendBC<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    doYVortTiltAvg<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    zeroTemArrays<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

    calczvorttilt<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    applyVortTendBC<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    doZVortTiltAvg<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    zeroTemArrays<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

    // Compute the baroclinic generation
    calcvortbaro<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

    // Do the SGS turbulence closure calculations
    doCalcDef<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    doGetTau<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    doCalcTurb<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

    zeroTemArrays<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    doTurbVort<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // apply the lower boundary condition to the horizontal
    // components of vorticity
    applyVortBC<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // Average the vorticity to the scalar grid using the temporary
    // arrays we allocated. After doing the averaging, we have to 
    // set the pointers to the temporary arrays as the new xvort,
    // yvort, and zvort, and set the old x/y/zvort arrays as the new
    // temporary arrays. Note: may have to zero those out in the future...
    doTurbVortAvg<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );

    zeroTemArrays<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk( cudaPeekAtLastError() );
    calczvortsolenoid<<<numBlocks, threadsPerBlock>>>(grid, data, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
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

        float pcl_x, pcl_y, pcl_z;
        float pcl_u, pcl_v, pcl_w;
        float uu1, vv1, ww1;
        float point[3];

        // loop over the number of time steps we are
        // integrating over
        float dt = 0.5;
        float dt2 = dt / 2.;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {

            // get the current values of various fields interpolated
            // to the parcel before we integrate using the RK2 step
            point[0] = parcels->xpos[PCL(tidx, parcel_id, totTime)];
            point[1] = parcels->ypos[PCL(tidx, parcel_id, totTime)];
            point[2] = parcels->zpos[PCL(tidx, parcel_id, totTime)];
            pcl_x = point[0];
            pcl_y = point[1];
            pcl_z = point[2];


            is_ugrd = true;
            is_vgrd = false;
            is_wgrd = false;
            pcl_u = interp3D(grid, data->u_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pcluturb = interp3D(grid, data->turbu_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            is_ugrd = false;
            is_vgrd = true;
            is_wgrd = false;
            pcl_v = interp3D(grid, data->v_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclvturb = interp3D(grid, data->turbv_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = true;
            pcl_w = interp3D(grid, data->w_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclwturb = interp3D(grid, data->turbw_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            // interpolate scalar values to the parcel point
            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = false;
            float pclxvort = interp3D(grid, data->xvort_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclyvort = interp3D(grid, data->yvort_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclzvort = interp3D(grid, data->zvort_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclxvorttilt = interp3D(grid, data->xvtilt_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclyvorttilt = interp3D(grid, data->yvtilt_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclzvorttilt = interp3D(grid, data->zvtilt_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclxvortstretch = interp3D(grid, data->xvstretch_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclyvortstretch = interp3D(grid, data->yvstretch_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclzvortstretch = interp3D(grid, data->zvstretch_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclxvortturb = interp3D(grid, data->turbxvort_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclyvortturb = interp3D(grid, data->turbyvort_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclzvortturb = interp3D(grid, data->turbzvort_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclxvortbaro = interp3D(grid, data->xvbaro_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclyvortbaro = interp3D(grid, data->yvbaro_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            float pclzvortsolenoid = interp3D(grid, data->zvort_solenoid_4d_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            parcels->pclu[PCL(tidx,   parcel_id, totTime)] = pcl_u;
            parcels->pclv[PCL(tidx,   parcel_id, totTime)] = pcl_v;
            parcels->pclw[PCL(tidx,   parcel_id, totTime)] = pcl_w;
            parcels->pcluturb[PCL(tidx,   parcel_id, totTime)] = pcluturb;
            parcels->pclvturb[PCL(tidx,   parcel_id, totTime)] = pclvturb;
            parcels->pclwturb[PCL(tidx,   parcel_id, totTime)] = pclwturb;

            // Store the vorticity in the parcel
            parcels->pclxvort[PCL(tidx, parcel_id, totTime)] = pclxvort;
            parcels->pclyvort[PCL(tidx, parcel_id, totTime)] = pclyvort;
            parcels->pclzvort[PCL(tidx, parcel_id, totTime)] = pclzvort;
            parcels->pclxvorttilt[PCL(tidx, parcel_id, totTime)] = pclxvorttilt;
            parcels->pclyvorttilt[PCL(tidx, parcel_id, totTime)] = pclyvorttilt;
            parcels->pclzvorttilt[PCL(tidx, parcel_id, totTime)] = pclzvorttilt;
            parcels->pclxvortstretch[PCL(tidx, parcel_id, totTime)] = pclxvortstretch;
            parcels->pclyvortstretch[PCL(tidx, parcel_id, totTime)] = pclyvortstretch;
            parcels->pclzvortstretch[PCL(tidx, parcel_id, totTime)] = pclzvortstretch;
            parcels->pclxvortturb[PCL(tidx, parcel_id, totTime)] = pclxvortturb;
            parcels->pclyvortturb[PCL(tidx, parcel_id, totTime)] = pclyvortturb;
            parcels->pclzvortturb[PCL(tidx, parcel_id, totTime)] = pclzvortturb;
            parcels->pclxvortbaro[PCL(tidx, parcel_id, totTime)] = pclxvortbaro;
            parcels->pclyvortbaro[PCL(tidx, parcel_id, totTime)] = pclyvortbaro;
            parcels->pclzvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclzvortsolenoid;

            // Now we use an RK2 scheme to integrate forward
            // in time. Values are interpolated to the parcel 
            // at the beginning of the next data time step. 
            for (int nkrp = 1; nkrp <= 2; ++nkrp) {        
                if (nkrp == 1) {
                    // integrate X position forward by the U wind
                    point[0] = pcl_x + pcl_u * dt * direct;
                    // integrate Y position forward by the V wind
                    point[1] = pcl_y + pcl_v * dt * direct;
                    // integrate Z position forward by the W wind
                    point[2] = pcl_z + pcl_w * dt * direct;
                    if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                        printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                            point[0], point[1], point[2], grid->xh[0], grid->yh[0], grid->zh[0], grid->xh[grid->NX-1], grid->yh[grid->NY-1], grid->zh[grid->NZ-1]);
                        return;
                    }
                    uu1 = pcl_u;
                    vv1 = pcl_v;
                    ww1 = pcl_w;
                }
                else {
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
                    point[0] = pcl_x + (pcl_u + uu1) * dt2 * direct;
                    // integrate Y position forward by the V wind
                    point[1] = pcl_y + (pcl_v + vv1) * dt2 * direct;
                    // integrate Z position forward by the W wind
                    point[2] = pcl_z + (pcl_w + ww1) * dt2 * direct;
                    if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                        printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                            point[0], point[1], point[2], grid->xh[0], grid->yh[0], grid->zh[0], grid->xh[grid->NX-1], grid->yh[grid->NY-1], grid->zh[grid->NZ-1]);
                        return;
                    }
                }
            } // end RK loop
            parcels->xpos[PCL(tidx+1, parcel_id, totTime)] = point[0]; 
            parcels->ypos[PCL(tidx+1, parcel_id, totTime)] = point[1];
            parcels->zpos[PCL(tidx+1, parcel_id, totTime)] = point[2];
        } // end time loop
    } // end index check
}

/*This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff*/
void cudaIntegrateParcels(datagrid *grid, integration_data *data, parcel_pos *parcels, int nT, int totTime, int direct) {

    int tStart, tEnd;
    tStart = 0;
    tEnd = nT;
    int NX, NY, NZ;
    // set the NX, NY, NZ
    // variables for calculations
    NX = grid->NX;
    NY = grid->NY;
    NZ = grid->NZ;


    // set the thread/block execution strategy for the kernels

    // Okay, so I think the last remaining issue might lie here. For some reason, some blocks 
    // must not be executing or something, seemingly related to the threadsPerBlock size. 
    // Changing to 4x4x4 fixed for xvort, but not yvort. I think we need to dynamically set
    // threadsPerBloc(x, y, z) based on the size of our grid at a given time step. 
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((int)ceil(NX/threadsPerBlock.x)+1, (int)ceil(NY/threadsPerBlock.y)+1, (int)ceil(NZ/threadsPerBlock.z)+1); 

    // we synchronize the device before doing anything to make sure all
    // array memory transfers have safely completed. This is probably 
    // unnecessary but I'm doing it anyways because overcaution never
    // goes wrong. Ever.
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // Calculate the three compionents of vorticity
    // and do the necessary averaging. This is a wrapper that
    // calls the necessary kernels and assigns the pointers
    // appropriately such that the "user" only has to call this method.
    doCalcVort(grid, data, tStart, tEnd, numBlocks, threadsPerBlock);
    

    // Calculate the vorticity forcing terms for each of the 3 components.
    // This is a wrapper that calls the necessary kernels to compute the
    // derivatives and average them back to the scalar grid where necessary. 
    doCalcVortTend(grid, data, tStart, tEnd, numBlocks, threadsPerBlock);
    // Before integrating the trajectories, George Bryan sets some below-grid/surface conditions 
    // that we need to consider. This handles applying those boundary conditions. 
    applyMomentumBC<<<numBlocks, threadsPerBlock>>>(data->u_4d_chunk, data->v_4d_chunk, data->w_4d_chunk, NX, NY, NZ, tStart, tEnd);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // integrate the parcels forward in time and interpolate
    // calculations to trajectories. 
    integrate<<<parcels->nParcels, 1>>>(grid, parcels, data, tStart, tEnd, totTime, direct);
    gpuErrchk(cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

}

#endif
