#include <iostream>
#include <stdio.h>
extern "C" {
#include <lofs-read.h>
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#include <lofs-limits.h>
#include <lofs-macros.h>
}
#include "../include/datastructs.h"
#include "../kernels/momentum.cu"
#include "../kernels/turb.cu"
#include "../kernels/vort.cu"
#include "../kernels/diff6.cu"
#include "interp.cu"
#ifndef INTEGRATE_CU
#define INTEGRATE_CU
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

// this is an error checking helper function for processes
// that run on the GPU. Without calling this, the GPU can
// fail to execute but the program won't crash or report it.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
       std::cout << cudaGetErrorString(code) << std::endl;
      if (abort) exit(code);
   }
}


//[>  Execute all of the required kernels on the GPU that are necessary for computing the 3
    //components of vorticity. The idea here is that we're building wrappers on wrappers to
    //simplify the process for the end user that just wants to calculate vorticity. This is
    //also a necessary adjustment because the tendency calculations will require multiple
    //steps, so transitioning this block of code as a proof of concept for how the programming
    //model should work. */
//void doCalcVort(grid *gd, mesh *msh, model_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t stream) {
    //// calculate the three compionents of vorticity
	//long bufidx;
	//int NX = gd->NX;
	//int NY = gd->NY;
	//int NZ = gd->NZ;
	//for ( int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcXvort<<<numBlocks, threadsPerBlock, 0, stream>>>(gd, &(data->vstag[bufidx]), &(data->wstag[bufidx]), &(data->tem1[bufidx]));
		//cuCalcYvort<<<numBlocks, threadsPerBlock, 0, stream>>>(gd, &(data->ustag[bufidx]), &(data->wstag[bufidx]), &(data->tem2[bufidx]));
		//cuCalcZvort<<<numBlocks, threadsPerBlock, 0, stream>>>(gd, &(data->ustag[bufidx]), &(data->vstag[bufidx]), &(data->tem3[bufidx]));
	//}

	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );
	//doVortAvg<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data->tem1, data->tem2, data->tem3, data->xvort, data->yvort, data->zvort, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );
	//zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );
//} 

//void doMomentumBud(datagrid *grid, model_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t stream) {
	//long bufidx;
	//int NX = grid->NX;
	//int NY = grid->NY;
	//int NZ = grid->NZ;

	//// Make sure we don't have any weird residual values
	//// in the temporary arrays
    //zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk( cudaStreamSynchronize(stream) );

	//// The scalars need to be computed and synchronized first
	//for ( int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcPipert<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->prespert[bufidx]), &(data->pipert[bufidx]));
		//cuCalcRf<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->rhopert[bufidx]), &(data->rhof[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));

	//for (int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcBuoy<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->thrhopert[bufidx]), &(data->buoy[bufidx]));
		//cuCalcPgradU<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->pgradu[bufidx]));
		//cuCalcPgradV<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->pgradv[bufidx]));
		//cuCalcPgradW<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->pgradw[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));

	//// These loops depend on temporary arrays so
	//// we have to be careful to not overwrite them

	//// compute momentum forcing on U from diffusion
	//for (int tidx = tStart; tidx < tEnd; ++tidx) { 
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

        //cuCalcDiffUXYZ<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->ustag[bufidx]), &(data->tem1[bufidx]), \
																  //&(data->tem2[bufidx]), &(data->tem3[bufidx]));
        //cuCalcDiff<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															//&(data->tem3[bufidx]), &(data->diffu[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));
    //zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream));

	//// compute momentum forcing on V from diffusion
	//for (int tidx = tStart; tidx < tEnd; ++tidx) { 
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

        //cuCalcDiffVXYZ<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->vstag[bufidx]), &(data->tem1[bufidx]), \
																  //&(data->tem2[bufidx]), &(data->tem3[bufidx]));
        //cuCalcDiff<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															//&(data->tem3[bufidx]), &(data->diffv[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));
    //zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream));

	//// compute momentum forcing on W from diffusion
	//for (int tidx = tStart; tidx < tEnd; ++tidx) { 
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

        //cuCalcDiffWXYZ<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->wstag[bufidx]), &(data->tem1[bufidx]), \
																  //&(data->tem2[bufidx]), &(data->tem3[bufidx]));
        //cuCalcDiff<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															//&(data->tem3[bufidx]), &(data->diffw[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));
    //zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream));


	//// Calculate the strain rate terms
	//for (int tidx = tStart; tidx < tEnd; ++tidx) { 
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		
		//cuCalcStrain<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->ustag[bufidx]), &(data->vstag[bufidx]), &(data->wstag[bufidx]), \
															   //&(data->rhopert[bufidx]), &(data->rhof[bufidx]), &(data->tem1[bufidx]),  \
															   //&(data->tem2[bufidx]), &(data->tem3[bufidx]), &(data->tem4[bufidx]), \
															   //&(data->tem5[bufidx]), &(data->tem6[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));
	//// Compute the stress terms from the strain rate
	//for (int tidx = tStart; tidx < tEnd; ++tidx) { 
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
        //cuGetTau<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->kmh[bufidx]), &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															//&(data->tem3[bufidx]), &(data->tem4[bufidx]), &(data->tem5[bufidx]), &(data->tem6[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));
	//// compute the momentum tendency due to turbulence closure
	//for (int tidx = tStart; tidx < tEnd; ++tidx) { 
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcTurb<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->tem1[bufidx]), &(data->tem2[bufidx]), &(data->tem3[bufidx]), \
															 //&(data->tem4[bufidx]), &(data->tem5[bufidx]), &(data->tem6[bufidx]), \
															 //&(data->rhopert[bufidx]), &(data->rhof[bufidx]), &(data->turbu[bufidx]), \
															 //&(data->turbv[bufidx]), &(data->turbw[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));
    //zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream));
//}

//void doCalcVortTend(datagrid *grid, model_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t stream) {
    //// get the io config from the user namelist
    //iocfg *io = data->io;

	//long bufidx;
	//int NX = grid->NX;
	//int NY = grid->NY;
	//int NZ = grid->NZ;

	//// The scalars need to be computed and synchronized first
	//for ( int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcPipert<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->prespert[bufidx]), &(data->pipert[bufidx]));
		//cuCalcRf<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->rhopert[bufidx]), &(data->rhof[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));

	//// Compute the budget terms for tilting of vorticity. First, we have to
	//// preprocess some derivatives on the staggered mesh into the temporary 
	//// arrays we have available, and then compute the tilting rate.
	//for (int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuPreXvortTilt<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->ustag[bufidx]), &(data->tem1[bufidx]), &(data->tem2[bufidx]));
		//cuPreYvortTilt<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->vstag[bufidx]), &(data->tem3[bufidx]), &(data->tem4[bufidx]));
		//cuPreZvortTilt<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->wstag[bufidx]), &(data->tem5[bufidx]), &(data->tem6[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));

	//// Now take the preprocessed derivatives and compute the tilting rate.
	//for (int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcXvortTilt<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->yvort[bufidx]), &(data->zvort[bufidx]), \
																  //&(data->tem1[bufidx]), &(data->tem2[bufidx]), &(data->xvtilt[bufidx]));
		//cuCalcYvortTilt<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->xvort[bufidx]), &(data->zvort[bufidx]), \
																  //&(data->tem3[bufidx]), &(data->tem4[bufidx]), &(data->yvtilt[bufidx]));
		//cuCalcZvortTilt<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->xvort[bufidx]), &(data->yvort[bufidx]), \
																  //&(data->tem5[bufidx]), &(data->tem6[bufidx]), &(data->zvtilt[bufidx]));
	//}
	//// synchronize and then clean out the tem arrays
	//gpuErrchk(cudaStreamSynchronize(stream));
	//zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream));
	//gpuErrchk( cudaPeekAtLastError() );

	//// Now do the budget terms that may or may not
	//// depend on the above scalars. These do not depend
	//// on temp arrays or each other and can be run
	//// in any order.
	//for (int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

		//// Compute the vorticity tendency due to stretching. These conveniently
		//// end up on the scalar grid, and no extra steps are required. This will
		//// compute the tendency for all 3 components of vorticity. 
		//cuCalcXvortStretch<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->vstag[bufidx]), &(data->wstag[bufidx]), \
																	  //&(data->xvort[bufidx]), &(data->xvstretch[bufidx]));
		//cuCalcYvortStretch<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->ustag[bufidx]), &(data->wstag[bufidx]), \
																	  //&(data->yvort[bufidx]), &(data->yvstretch[bufidx]));
		//cuCalcZvortStretch<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->ustag[bufidx]), &(data->vstag[bufidx]), \
																	  //&(data->zvort[bufidx]), &(data->zvstretch[bufidx]));
 
         //cuCalcXvortBaro<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->thrhopert[bufidx]), &(data->xvort_baro[bufidx]));
         //cuCalcYvortBaro<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->thrhopert[bufidx]), &(data->yvort_baro[bufidx]));

         //cuCalcXvortSolenoid<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->xvort_solenoid[bufidx]));
         //cuCalcYvortSolenoid<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->yvort_solenoid[bufidx]));
         //cuCalcZvortSolenoid<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->zvort_solenoid[bufidx]));
 
	//}
	//gpuErrchk(cudaStreamSynchronize(stream));
	//gpuErrchk(cudaPeekAtLastError());

	//// Have we already computed the momentum budgets?
	//// If not, calculate them for our diffusion and turbulence terms.
	//if (!io->output_momentum_budget) {
		//doMomentumBud(grid, data, tStart, tEnd, numBlocks, threadsPerBlock, stream);
		//zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
		//gpuErrchk(cudaStreamSynchronize(stream));
		//gpuErrchk( cudaPeekAtLastError() );
	//}

	//// compute vorticity due to momentum diffusion
	//for ( int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcXvort<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->diffv[bufidx]), &(data->diffw[bufidx]), &(data->tem1[bufidx]));
		//cuCalcYvort<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->diffu[bufidx]), &(data->diffw[bufidx]), &(data->tem2[bufidx]));
		//cuCalcZvort<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->diffu[bufidx]), &(data->diffv[bufidx]), &(data->tem3[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );
	//doVortAvg<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data->tem1, data->tem2, data->tem3, data->diffxvort, data->diffyvort, data->diffzvort, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );
	//zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );

	//// compute vorticity due to momentum turbulence
	//for ( int tidx = tStart; tidx < tEnd; ++tidx) {
        //bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		//cuCalcXvort<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->turbv[bufidx]), &(data->turbw[bufidx]), &(data->tem1[bufidx]));
		//cuCalcYvort<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->turbu[bufidx]), &(data->turbw[bufidx]), &(data->tem2[bufidx]));
		//cuCalcZvort<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, &(data->turbu[bufidx]), &(data->turbv[bufidx]), &(data->tem3[bufidx]));
	//}
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );
	//doVortAvg<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data->tem1, data->tem2, data->tem3, data->turbxvort, data->turbyvort, data->turbzvort, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );
	//zeroTemArrays<<<numBlocks, threadsPerBlock, 0, stream>>>(grid, data, tStart, tEnd);
	//gpuErrchk(cudaStreamSynchronize(stream) );
	//gpuErrchk( cudaPeekAtLastError() );

//}

//__global__ void integrate(datagrid *grid, parcel_pos *parcels, model_data *data, \
                          //int tStart, int tEnd, int totTime, int direct) {

	////int parcel_id = blockIdx.x;
    //int parcel_id = blockIdx.x * blockDim.x + threadIdx.x;

    //// safety check to make sure our thread index doesn't
    //// go out of our array bounds
    //if (parcel_id < parcels->nParcels) {
        //bool is_ugrd = false;
        //bool is_vgrd = false;
        //bool is_wgrd = false;

        //float pcl_x, pcl_y, pcl_z;
        //float pcl_u, pcl_v, pcl_w;
        //float uu1, vv1, ww1;
        //float point[3];

        //// loop over the number of time steps we are
        //// integrating over
        //float dt = grid->dt; 
        //float dt2 = dt / 2.;
        //for (int tidx = tStart; tidx < tEnd; ++tidx) {

            //// get the current values of various fields interpolated
            //// to the parcel before we integrate using the RK2 step
            //point[0] = parcels->xpos[PCL(tidx, parcel_id, totTime)];
            //point[1] = parcels->ypos[PCL(tidx, parcel_id, totTime)];
            //point[2] = parcels->zpos[PCL(tidx, parcel_id, totTime)];
            //pcl_x = point[0];
            //pcl_y = point[1];
            //pcl_z = point[2];
            //if (( pcl_x > xf(grid->NX-4) ) || ( pcl_y > yf(grid->NY-4) ) || ( pcl_z > zf(grid->NZ-4) ) \
             //|| ( pcl_x < xf(0) )        || ( pcl_y < yf(0) )        || ( pcl_z < 0. ) ) {
                //break;
            //}


            //is_ugrd = true;
            //is_vgrd = false;
            //is_wgrd = false;
            //pcl_u = interp3D(grid, data->ustag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            //is_ugrd = false;
            //is_vgrd = true;
            //is_wgrd = false;
            //pcl_v = interp3D(grid, data->vstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

            //is_ugrd = false;
            //is_vgrd = false;
            //is_wgrd = true;
            //pcl_w = interp3D(grid, data->wstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);
            //parcels->pclu[PCL(tidx,   parcel_id, totTime)] = pcl_u;
            //parcels->pclv[PCL(tidx,   parcel_id, totTime)] = pcl_v;
            //parcels->pclw[PCL(tidx,   parcel_id, totTime)] = pcl_w;

            //// Now we use an RK2 scheme to integrate forward
            //// in time. Values are interpolated to the parcel 
            //// at the beginning of the next data time step. 
            //for (int nkrp = 1; nkrp <= 2; ++nkrp) {        
                //if (nkrp == 1) {
                    //// integrate X position forward by the U wind
                    //point[0] = pcl_x + pcl_u * dt * direct;
                    //// integrate Y position forward by the V wind
                    //point[1] = pcl_y + pcl_v * dt * direct;
                    //// integrate Z position forward by the W wind
                    //point[2] = pcl_z + pcl_w * dt * direct;
                    //if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                        //printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                            //point[0], point[1], point[2], xh(0), yh(0), zh(0), xh(grid->NX-1), yh(grid->NY-1), zh(grid->NZ-1));
                        //return;
                    //}
                    //uu1 = pcl_u;
                    //vv1 = pcl_v;
                    //ww1 = pcl_w;
                //}
                //else {
                    //is_ugrd = true;
                    //is_vgrd = false;
                    //is_wgrd = false;
                    //pcl_u = interp3D(grid, data->ustag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

                    //is_ugrd = false;
                    //is_vgrd = true;
                    //is_wgrd = false;
                    //pcl_v = interp3D(grid, data->vstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

                    //is_ugrd = false;
                    //is_vgrd = false;
                    //is_wgrd = true;
                    //pcl_w = interp3D(grid, data->wstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

                    //// integrate X position forward by the U wind
                    //point[0] = pcl_x + (pcl_u + uu1) * dt2 * direct;
                    //// integrate Y position forward by the V wind
                    //point[1] = pcl_y + (pcl_v + vv1) * dt2 * direct;
                    //// integrate Z position forward by the W wind
                    //point[2] = pcl_z + (pcl_w + ww1) * dt2 * direct;
                    //if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                        //printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                            //point[0], point[1], point[2], xh(0), yh(0), zh(0), xh(grid->NX-1), yh(grid->NY-1), zh(grid->NZ-1));
                        //return;
                    //}
                //}
            //} // end RK loop

            //parcels->xpos[PCL(tidx+1, parcel_id, totTime)] = point[0]; 
            //parcels->ypos[PCL(tidx+1, parcel_id, totTime)] = point[1];
            //parcels->zpos[PCL(tidx+1, parcel_id, totTime)] = point[2];
        //} // end time loop
    //} // end index check
//}

//__global__ void parcel_interp(datagrid *grid, parcel_pos *parcels, model_data *data, \
                          //int tStart, int tEnd, int totTime, int direct) {

	////int parcel_id = blockIdx.x;
    //int parcel_id = blockIdx.x * blockDim.x + threadIdx.x;
    //// get the io config from the user namelist
    //iocfg *io = parcels->io;

    //// safety check to make sure our thread index doesn't
    //// go out of our array bounds
    //if (parcel_id < parcels->nParcels) {
        //bool is_ugrd = false;
        //bool is_vgrd = false;
        //bool is_wgrd = false;

        //float point[3];

        //// loop over the number of time steps we are
        //// integrating over
        //for (int tidx = tStart; tidx < tEnd; ++tidx) {
            //point[0] = parcels->xpos[PCL(tidx, parcel_id, totTime)];
            //point[1] = parcels->ypos[PCL(tidx, parcel_id, totTime)];
            //point[2] = parcels->zpos[PCL(tidx, parcel_id, totTime)];
            //if (io->output_kmh) {
                //is_ugrd = false;
                //is_vgrd = false;
                //is_wgrd = true;
                //float pclkmh = interp3D(grid, data->kmh, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclkmh[PCL(tidx, parcel_id, totTime)] = pclkmh;
            //}

            //if (io->output_momentum_budget) {
                //is_ugrd = true;
                //is_vgrd = false;
                //is_wgrd = false;
                //float pclupgrad = interp3D(grid, data->pgradu, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pcluturb = interp3D(grid, data->turbu, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pcludiff = interp3D(grid, data->diffu, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //is_ugrd = false;
                //is_vgrd = true;
                //is_wgrd = false;
                //float pclvpgrad = interp3D(grid, data->pgradv, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclvturb = interp3D(grid, data->turbv, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclvdiff = interp3D(grid, data->diffv, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //is_ugrd = false;
                //is_vgrd = false;
                //is_wgrd = true;
                //float pclwpgrad = interp3D(grid, data->pgradw, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclwturb = interp3D(grid, data->turbw, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclwdiff = interp3D(grid, data->diffw, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclbuoy = interp3D(grid, data->buoy, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclupgrad[PCL(tidx,   parcel_id, totTime)] = pclupgrad;
                //parcels->pclvpgrad[PCL(tidx,   parcel_id, totTime)] = pclvpgrad;
                //parcels->pclwpgrad[PCL(tidx,   parcel_id, totTime)] = pclwpgrad;
                //parcels->pcluturb[PCL(tidx,   parcel_id, totTime)] = pcluturb;
                //parcels->pclvturb[PCL(tidx,   parcel_id, totTime)] = pclvturb;
                //parcels->pclwturb[PCL(tidx,   parcel_id, totTime)] = pclwturb;
                //parcels->pcludiff[PCL(tidx,   parcel_id, totTime)] = pcludiff;
                //parcels->pclvdiff[PCL(tidx,   parcel_id, totTime)] = pclvdiff;
                //parcels->pclwdiff[PCL(tidx,   parcel_id, totTime)] = pclwdiff;
                //parcels->pclbuoy[PCL(tidx,   parcel_id, totTime)] = pclbuoy;
            //}


            //// interpolate scalar values to the parcel point
            //is_ugrd = false;
            //is_vgrd = false;
            //is_wgrd = false;
            //if (io->output_vorticity_budget || io->output_xvort) {
                //float pclxvort = interp3D(grid, data->xvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclxvort[PCL(tidx, parcel_id, totTime)] = pclxvort;
            //}
            //if (io->output_vorticity_budget || io->output_yvort) {
                //float pclyvort = interp3D(grid, data->yvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclyvort[PCL(tidx, parcel_id, totTime)] = pclyvort;
            //}
            //if (io->output_vorticity_budget || io->output_zvort) {
                //float pclzvort = interp3D(grid, data->zvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclzvort[PCL(tidx, parcel_id, totTime)] = pclzvort;
            //}
            //if (io->output_vorticity_budget) {
                //float pclxvorttilt = interp3D(grid, data->xvtilt, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclyvorttilt = interp3D(grid, data->yvtilt, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclzvorttilt = interp3D(grid, data->zvtilt, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclxvortstretch = interp3D(grid, data->xvstretch, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclyvortstretch = interp3D(grid, data->yvstretch, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclzvortstretch = interp3D(grid, data->zvstretch, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclxvortturb = interp3D(grid, data->turbxvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclyvortturb = interp3D(grid, data->turbyvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclzvortturb = interp3D(grid, data->turbzvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclxvortdiff = interp3D(grid, data->diffxvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclyvortdiff = interp3D(grid, data->diffyvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclzvortdiff = interp3D(grid, data->diffzvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclxvortbaro = interp3D(grid, data->xvort_baro, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclyvortbaro = interp3D(grid, data->yvort_baro, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclxvortsolenoid = interp3D(grid, data->xvort_solenoid, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclyvortsolenoid = interp3D(grid, data->yvort_solenoid, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //float pclzvortsolenoid = interp3D(grid, data->zvort_solenoid, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //// Store the vorticity in the parcel
                //parcels->pclxvorttilt[PCL(tidx, parcel_id, totTime)] = pclxvorttilt;
                //parcels->pclyvorttilt[PCL(tidx, parcel_id, totTime)] = pclyvorttilt;
                //parcels->pclzvorttilt[PCL(tidx, parcel_id, totTime)] = pclzvorttilt;
                //parcels->pclxvortstretch[PCL(tidx, parcel_id, totTime)] = pclxvortstretch;
                //parcels->pclyvortstretch[PCL(tidx, parcel_id, totTime)] = pclyvortstretch;
                //parcels->pclzvortstretch[PCL(tidx, parcel_id, totTime)] = pclzvortstretch;
                //parcels->pclxvortturb[PCL(tidx, parcel_id, totTime)] = pclxvortturb;
                //parcels->pclyvortturb[PCL(tidx, parcel_id, totTime)] = pclyvortturb;
                //parcels->pclzvortturb[PCL(tidx, parcel_id, totTime)] = pclzvortturb;
                //parcels->pclxvortdiff[PCL(tidx, parcel_id, totTime)] = pclxvortdiff;
                //parcels->pclyvortdiff[PCL(tidx, parcel_id, totTime)] = pclyvortdiff;
                //parcels->pclzvortdiff[PCL(tidx, parcel_id, totTime)] = pclzvortdiff;
                //parcels->pclxvortbaro[PCL(tidx, parcel_id, totTime)] = pclxvortbaro;
                //parcels->pclyvortbaro[PCL(tidx, parcel_id, totTime)] = pclyvortbaro;
                //parcels->pclxvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclxvortsolenoid;
                //parcels->pclyvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclyvortsolenoid;
                //parcels->pclzvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclzvortsolenoid;
            //}

            //// Now do the scalars
            //if (io->output_ppert) {
                //float pclppert = interp3D(grid, data->prespert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclppert[PCL(tidx, parcel_id, totTime)] = pclppert;
            //}
            //if (io->output_qvpert) {
                //float pclqvpert = interp3D(grid, data->qvpert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclqvpert[PCL(tidx, parcel_id, totTime)] = pclqvpert;
            //}
            //if (io->output_rhopert) {
                //float pclrhopert = interp3D(grid, data->rhopert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclrhopert[PCL(tidx, parcel_id, totTime)] = pclrhopert;
            //}
            //if (io->output_thetapert) {
                //float pclthetapert = interp3D(grid, data->thetapert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclthetapert[PCL(tidx, parcel_id, totTime)] = pclthetapert;
            //}
            //if (io->output_thrhopert) {
                //float pclthrhopert = interp3D(grid, data->thrhopert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclthrhopert[PCL(tidx, parcel_id, totTime)] = pclthrhopert;
            //}

            //if (io->output_pbar) {
                //float pclpbar = interp1D(grid, grid->p0, point[2], is_wgrd, tidx);
                //parcels->pclpbar[PCL(tidx, parcel_id, totTime)] = pclpbar;
            //}
            //if (io->output_qvbar) {
                //float pclqvbar = interp1D(grid, grid->qv0, point[2], is_wgrd, tidx);
                //parcels->pclqvbar[PCL(tidx, parcel_id, totTime)] = pclqvbar;
            //}
            //if (io->output_rhobar) {
                //float pclrhobar = interp1D(grid, grid->rho0, point[2], is_wgrd, tidx);
                //parcels->pclrhobar[PCL(tidx, parcel_id, totTime)] = pclrhobar;
            //}
            //if (io->output_thetabar) {
                //float pclthetabar = interp1D(grid, grid->th0, point[2], is_wgrd, tidx);
                //parcels->pclthetabar[PCL(tidx, parcel_id, totTime)] = pclthetabar;
            //}
            //if (io->output_rhobar) {
                //float pclthrhobar = interp1D(grid, grid->th0, point[2], is_wgrd, tidx);
                //parcels->pclthrhobar[PCL(tidx, parcel_id, totTime)] = pclthrhobar;
            //}

            //if (io->output_qc) {
                //float pclqc = interp3D(grid, data->qc, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclqc[PCL(tidx, parcel_id, totTime)] = pclqc;
            //}
            //if (io->output_qi) {
                //float pclqi = interp3D(grid, data->qi, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclqi[PCL(tidx, parcel_id, totTime)] = pclqi;
            //}
            //if (io->output_qs) {
                //float pclqs = interp3D(grid, data->qs, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclqs[PCL(tidx, parcel_id, totTime)] = pclqs;
            //}
            //if (io->output_qg) {
                //float pclqg = interp3D(grid, data->qg, point, is_ugrd, is_vgrd, is_wgrd, tidx);
                //parcels->pclqg[PCL(tidx, parcel_id, totTime)] = pclqg;
            //}
        //}
    //}
//}

//[>This function handles allocating memory on the GPU, transferring the CPU
//arrays to GPU global memory, calling the integrate GPU kernel, and then
//updating the position vectors with the new stuff*/
//void cudaIntegrateParcels(datagrid *grid, model_data *data, parcel_pos *parcels, int nT, int totTime, int direct) {

    //int tStart, tEnd;
    //tStart = 0;
    //tEnd = nT;
    //int NX, NY, NZ;
    //// set the NX, NY, NZ
    //// variables for calculations
    //NX = grid->NX;
    //NY = grid->NY;
    //NZ = grid->NZ;
    //iocfg *io = parcels->io;

    //cudaStream_t calStream;
    //cudaStream_t intStream;
    //cudaStreamCreate(&calStream);
    //cudaStreamCreate(&intStream);


    //// set the thread/block execution strategy for the kernels
    //dim3 threadsPerBlock(256, 1, 1);
    //dim3 numBlocks((int)ceil(NX+2/threadsPerBlock.x)+1, (int)ceil(NY+2/threadsPerBlock.y)+1, (int)ceil(NZ+1/threadsPerBlock.z)+1); 

    //if (io->output_momentum_budget) doMomentumBud(grid, data, tStart, tEnd, numBlocks, threadsPerBlock, calStream);
    //// Calculate the three compionents of vorticity
    //// and do the necessary averaging. This is a wrapper that
    //// calls the necessary kernels and assigns the pointers
    //// appropriately such that the "user" only has to call this method.
    //if (io->output_xvort || io->output_yvort || io->output_zvort || io->output_vorticity_budget) {
        //doCalcVort(grid, data, tStart, tEnd, numBlocks, threadsPerBlock, calStream);
    //}
    

    //// Calculate the vorticity forcing terms for each of the 3 components.
    //// This is a wrapper that calls the necessary kernels to compute the
    //// derivatives and average them back to the scalar grid where necessary. 
    //if (io->output_vorticity_budget) doCalcVortTend(grid, data, tStart, tEnd, numBlocks, threadsPerBlock, calStream);


    //// Before integrating the trajectories, George Bryan sets some below-grid/surface conditions 
    //// that we need to consider. This handles applying those boundary conditions. 
    ////applyMomentumBC<<<numBlocks, threadsPerBlock>>>(data->ustag, data->vstag, data->wstag, NX, NY, NZ, tStart, tEnd);
    ////gpuErrchk(cudaDeviceSynchronize() );
    ////gpuErrchk( cudaPeekAtLastError() );

    //// integrate the parcels forward in time and interpolate
    //// calculations to trajectories. 
    //int nThreads = 256;
    //int nPclBlocks = int(parcels->nParcels / nThreads) + 1;
    //integrate<<<nPclBlocks, nThreads, 0, intStream>>>(grid, parcels, data, tStart, tEnd, totTime, direct);
    //gpuErrchk(cudaDeviceSynchronize());
    //gpuErrchk( cudaPeekAtLastError() );

    //parcel_interp<<<nPclBlocks, nThreads, 0, intStream>>>(grid, parcels, data, tStart, tEnd, totTime, direct);
    //gpuErrchk(cudaDeviceSynchronize());
    //gpuErrchk( cudaPeekAtLastError() );
//}
#endif

