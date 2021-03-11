#include <iostream>
#include <stdio.h>
#include <chrono>
extern "C" {
#include <lofs-read.h>
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#include <lofs-limits.h>
#include <lofs-macros.h>
}
#include "../include/datastructs.h"
#include "../include/constants.h"
#include "../include/prefetch.h"
#include "../io/prefetch.cu"
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

void zeroTemArrays(grid *gd, model_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock) {
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStream_t stream4;
	cudaStream_t stream5;
	cudaStream_t stream6;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	cudaStreamCreate(&stream5);
	cudaStreamCreate(&stream6);
	zeroTemArray<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, data->tem1, tStart, tEnd);
	zeroTemArray<<<numBlocks, threadsPerBlock, 0, stream2>>>(gd, data->tem2, tStart, tEnd);
	zeroTemArray<<<numBlocks, threadsPerBlock, 0, stream3>>>(gd, data->tem3, tStart, tEnd);
	zeroTemArray<<<numBlocks, threadsPerBlock, 0, stream4>>>(gd, data->tem4, tStart, tEnd);
	zeroTemArray<<<numBlocks, threadsPerBlock, 0, stream5>>>(gd, data->tem5, tStart, tEnd);
	zeroTemArray<<<numBlocks, threadsPerBlock, 0, stream6>>>(gd, data->tem6, tStart, tEnd);
	cudaDeviceSynchronize();
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);
	cudaStreamDestroy(stream5);
	cudaStreamDestroy(stream6);
}

	/* Execute all of the required kernels on the GPU that are necessary for computing the 3
    * components of vorticity. The idea here is that we're building wrappers on wrappers to
    * simplify the process for the end user that just wants to calculate vorticity. This is
    * also a necessary adjustment because the tendency calculations will require multiple
    * steps, so transitioning this block of code as a proof of concept for how the programming
    * model should work. */
void doCalcVort(grid *gd, mesh *msh, model_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock) {
	// calculate the three compionents of vorticity
	long bufidx;
	int NX = gd->NX;
	int NY = gd->NY;
	int NZ = gd->NZ;
	cudaStream_t xvort_stream;
	cudaStream_t yvort_stream;
	cudaStream_t zvort_stream;
	cudaStreamCreate(&xvort_stream);
	cudaStreamCreate(&yvort_stream);
	cudaStreamCreate(&zvort_stream);
	for ( int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcXvort<<<numBlocks, threadsPerBlock, 0, xvort_stream>>>(gd, msh, &(data->vstag[bufidx]), &(data->wstag[bufidx]), &(data->tem1[bufidx]));
		cuCalcYvort<<<numBlocks, threadsPerBlock, 0, yvort_stream>>>(gd, msh, &(data->ustag[bufidx]), &(data->wstag[bufidx]), &(data->tem2[bufidx]));
		cuCalcZvort<<<numBlocks, threadsPerBlock, 0, zvort_stream>>>(gd, msh, &(data->ustag[bufidx]), &(data->vstag[bufidx]), &(data->tem3[bufidx]));
	}

	gpuErrchk(cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	doVortAvg<<<numBlocks, threadsPerBlock>>>(gd, data->tem1, data->tem2, data->tem3, data->xvort, data->yvort, data->zvort, tStart, tEnd);
	gpuErrchk(cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);
	gpuErrchk(cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	cudaStreamDestroy(xvort_stream);
	cudaStreamDestroy(yvort_stream);
	cudaStreamDestroy(zvort_stream);
} 

void doMomentumBud(grid *gd, mesh *msh, sounding *snd, model_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock) {
	long bufidx;
	int NX = gd->NX;
	int NY = gd->NY;
	int NZ = gd->NZ;

	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStream_t stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);

	// The scalars need to be computed and synchronized first
	for ( int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcPipert<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, snd, &(data->prespert[bufidx]), &(data->pipert[bufidx]));
		cuCalcRf<<<numBlocks, threadsPerBlock, 0, stream2>>>(gd, msh, snd, &(data->rhopert[bufidx]), &(data->rhof[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());

	for (int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcBuoy<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, msh, snd, &(data->thrhopert[bufidx]), &(data->buoy[bufidx]));
		cuCalcPgradU<<<numBlocks, threadsPerBlock, 0, stream2>>>(gd, msh, snd, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->pgradu[bufidx]));
		cuCalcPgradV<<<numBlocks, threadsPerBlock, 0, stream3>>>(gd, msh, snd, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->pgradv[bufidx]));
		cuCalcPgradW<<<numBlocks, threadsPerBlock, 0, stream4>>>(gd, msh, snd, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->pgradw[bufidx]));
	}
	//gpuErrchk(cudaDeviceSynchronize());

	// These loops depend on temporary arrays so
	// we have to be careful to not overwrite them

	// compute momentum forcing on U from diffusion
	for (int tidx = tStart; tidx < tEnd; ++tidx) { 
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

		cuCalcDiffUXYZ<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, msh, snd, &(data->ustag[bufidx]), &(data->tem1[bufidx]), \
																  &(data->tem2[bufidx]), &(data->tem3[bufidx]));
		cuCalcDiff<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, msh, snd, &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															&(data->tem3[bufidx]), &(data->diffu[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);

	// compute momentum forcing on V from diffusion
	for (int tidx = tStart; tidx < tEnd; ++tidx) { 
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

		cuCalcDiffVXYZ<<<numBlocks, threadsPerBlock, 0, stream2>>>(gd, msh, snd, &(data->vstag[bufidx]), &(data->tem1[bufidx]), \
																  &(data->tem2[bufidx]), &(data->tem3[bufidx]));
		cuCalcDiff<<<numBlocks, threadsPerBlock, 0, stream2>>>(gd, msh, snd, &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															&(data->tem3[bufidx]), &(data->diffv[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);

	// compute momentum forcing on W from diffusion
	for (int tidx = tStart; tidx < tEnd; ++tidx) { 
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

		cuCalcDiffWXYZ<<<numBlocks, threadsPerBlock, 0, stream3>>>(gd, msh, snd, &(data->wstag[bufidx]), &(data->tem1[bufidx]), \
																  &(data->tem2[bufidx]), &(data->tem3[bufidx]));
		cuCalcDiff<<<numBlocks, threadsPerBlock, 0, stream3>>>(gd, msh, snd, &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															&(data->tem3[bufidx]), &(data->diffw[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);


	// Calculate the strain rate terms
	for (int tidx = tStart; tidx < tEnd; ++tidx) { 
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		
		cuCalcStrain<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, msh, snd, &(data->ustag[bufidx]), &(data->vstag[bufidx]), &(data->wstag[bufidx]), \
															   &(data->rhopert[bufidx]), &(data->rhof[bufidx]), &(data->tem1[bufidx]),  \
															   &(data->tem2[bufidx]), &(data->tem3[bufidx]), &(data->tem4[bufidx]), \
															   &(data->tem5[bufidx]), &(data->tem6[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());
	// Compute the stress terms from the strain rate
	for (int tidx = tStart; tidx < tEnd; ++tidx) { 
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuGetTau<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, msh, snd, &(data->kmh[bufidx]), &(data->tem1[bufidx]), &(data->tem2[bufidx]), \
															&(data->tem3[bufidx]), &(data->tem4[bufidx]), &(data->tem5[bufidx]), &(data->tem6[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());
	// compute the momentum tendency due to turbulence closure
	for (int tidx = tStart; tidx < tEnd; ++tidx) { 
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcTurb<<<numBlocks, threadsPerBlock, 0, stream1>>>(gd, msh, snd, &(data->tem1[bufidx]), &(data->tem2[bufidx]), &(data->tem3[bufidx]), \
															 &(data->tem4[bufidx]), &(data->tem5[bufidx]), &(data->tem6[bufidx]), \
															 &(data->rhopert[bufidx]), &(data->rhof[bufidx]), &(data->turbu[bufidx]), \
															 &(data->turbv[bufidx]), &(data->turbw[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);
}

void doCalcVortTend(grid *gd, mesh *msh, sounding *snd, model_data *data, int tStart, int tEnd, dim3 numBlocks, dim3 threadsPerBlock) {
	// get the io config from the user namelist
	iocfg *io = data->io;

	long bufidx;
	int NX = gd->NX;
	int NY = gd->NY;
	int NZ = gd->NZ;

	// The scalars need to be computed and synchronized first
	for ( int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcPipert<<<numBlocks, threadsPerBlock>>>(gd, snd, &(data->prespert[bufidx]), &(data->pipert[bufidx]));
		cuCalcRf<<<numBlocks, threadsPerBlock>>>(gd, msh, snd, &(data->rhopert[bufidx]), &(data->rhof[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());

	// Compute the budget terms for tilting of vorticity. First, we have to
	// preprocess some derivatives on the staggered mesh into the temporary 
	// arrays we have available, and then compute the tilting rate.
	for (int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuPreXvortTilt<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->ustag[bufidx]), &(data->tem1[bufidx]), &(data->tem2[bufidx]));
		cuPreYvortTilt<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->vstag[bufidx]), &(data->tem3[bufidx]), &(data->tem4[bufidx]));
		cuPreZvortTilt<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->wstag[bufidx]), &(data->tem5[bufidx]), &(data->tem6[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize());

	// Now take the preprocessed derivatives and compute the tilting rate.
	for (int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcXvortTilt<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->yvort[bufidx]), &(data->zvort[bufidx]), \
																  &(data->tem1[bufidx]), &(data->tem2[bufidx]), &(data->xvtilt[bufidx]));
		cuCalcYvortTilt<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->xvort[bufidx]), &(data->zvort[bufidx]), \
																  &(data->tem3[bufidx]), &(data->tem4[bufidx]), &(data->yvtilt[bufidx]));
		cuCalcZvortTilt<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->xvort[bufidx]), &(data->yvort[bufidx]), \
																  &(data->tem5[bufidx]), &(data->tem6[bufidx]), &(data->zvtilt[bufidx]));
	}
	// synchronize and then clean out the tem arrays
	gpuErrchk(cudaDeviceSynchronize());
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);
	gpuErrchk( cudaPeekAtLastError() );

	// Now do the budget terms that may or may not
	// depend on the above scalars. These do not depend
	// on temp arrays or each other and can be run
	// in any order.
	for (int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);

		// Compute the vorticity tendency due to stretching. These conveniently
		// end up on the scalar grid, and no extra steps are required. This will
		// compute the tendency for all 3 components of vorticity. 
		cuCalcXvortStretch<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->vstag[bufidx]), &(data->wstag[bufidx]), \
																	  &(data->xvort[bufidx]), &(data->xvstretch[bufidx]));
		cuCalcYvortStretch<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->ustag[bufidx]), &(data->wstag[bufidx]), \
																	  &(data->yvort[bufidx]), &(data->yvstretch[bufidx]));
		cuCalcZvortStretch<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->ustag[bufidx]), &(data->vstag[bufidx]), \
																	  &(data->zvort[bufidx]), &(data->zvstretch[bufidx]));
 
		 cuCalcXvortBaro<<<numBlocks, threadsPerBlock>>>(gd, msh, snd, &(data->thrhopert[bufidx]), &(data->xvort_baro[bufidx]));
		 cuCalcYvortBaro<<<numBlocks, threadsPerBlock>>>(gd, msh, snd, &(data->thrhopert[bufidx]), &(data->yvort_baro[bufidx]));

		 cuCalcXvortSolenoid<<<numBlocks, threadsPerBlock>>>(gd, msh, snd, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->xvort_solenoid[bufidx]));
		 cuCalcYvortSolenoid<<<numBlocks, threadsPerBlock>>>(gd, msh, snd, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->yvort_solenoid[bufidx]));
		 cuCalcZvortSolenoid<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->pipert[bufidx]), &(data->thrhopert[bufidx]), &(data->zvort_solenoid[bufidx]));
 
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	// Have we already computed the momentum budgets?
	// If not, calculate them for our diffusion and turbulence terms.
	if (!io->output_momentum_budget) {
		doMomentumBud(gd, msh, snd, data, tStart, tEnd, numBlocks, threadsPerBlock);
		//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);
		gpuErrchk( cudaPeekAtLastError() );
	}

	// compute vorticity due to momentum diffusion
	for ( int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcXvort<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->diffv[bufidx]), &(data->diffw[bufidx]), &(data->tem1[bufidx]));
		cuCalcYvort<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->diffu[bufidx]), &(data->diffw[bufidx]), &(data->tem2[bufidx]));
		cuCalcZvort<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->diffu[bufidx]), &(data->diffv[bufidx]), &(data->tem3[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	doVortAvg<<<numBlocks, threadsPerBlock>>>(gd, data->tem1, data->tem2, data->tem3, data->diffxvort, data->diffyvort, data->diffzvort, tStart, tEnd);
	gpuErrchk(cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);
	gpuErrchk( cudaPeekAtLastError() );

	// compute vorticity due to momentum turbulence
	for ( int tidx = tStart; tidx < tEnd; ++tidx) {
		bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
		cuCalcXvort<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->turbv[bufidx]), &(data->turbw[bufidx]), &(data->tem1[bufidx]));
		cuCalcYvort<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->turbu[bufidx]), &(data->turbw[bufidx]), &(data->tem2[bufidx]));
		cuCalcZvort<<<numBlocks, threadsPerBlock>>>(gd, msh, &(data->turbu[bufidx]), &(data->turbv[bufidx]), &(data->tem3[bufidx]));
	}
	gpuErrchk(cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	doVortAvg<<<numBlocks, threadsPerBlock>>>(gd, data->tem1, data->tem2, data->tem3, data->turbxvort, data->turbyvort, data->turbzvort, tStart, tEnd);
	gpuErrchk(cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	//zeroTemArrays(gd, data, tStart, tEnd, numBlocks, threadsPerBlock);
	gpuErrchk( cudaPeekAtLastError() );

}

__global__ void integrate(grid *gd, mesh *msh, sounding *snd, parcel_pos *parcels, model_data *data, \
						  float fill, int tStart, int tEnd, int totTime, int direct) {

	//int parcel_id = blockIdx.x;
	int parcel_id = blockIdx.x * blockDim.x + threadIdx.x;

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
		float dt = msh->dt; 
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
			if ((point[2] > zf(gd->NZ)) || (point[0] > xf(gd->NX)) || (point[0] > yf(gd->NY)) || \
			    (point[2] < zf(0))      || (point[0] < xf(0))       || (point[1] < yf(0)) || (point[2] == fill)) {
				//printf("%f %f %f\n%f %f %f\n%f %f %f\n", point[0], point[1], point[2], xf(0), yf(0), zf(0), xf(gd->NX), yf(gd->NY), zf(gd->NZ));
				point[0] = fill; 
				point[1] = fill; 
				point[2] = fill; 
				pcl_u = fill;
				pcl_v = fill;
				pcl_w = fill;

				parcels->xpos[PCL(tidx+1, parcel_id, totTime)] = point[0]; 
				parcels->ypos[PCL(tidx+1, parcel_id, totTime)] = point[1];
				parcels->zpos[PCL(tidx+1, parcel_id, totTime)] = point[2];
				parcels->pclu[PCL(tidx,   parcel_id, totTime)] = pcl_u;
				parcels->pclv[PCL(tidx,   parcel_id, totTime)] = pcl_v;
				parcels->pclw[PCL(tidx,   parcel_id, totTime)] = pcl_w;
				continue;
			}

			is_ugrd = true;
			is_vgrd = false;
			is_wgrd = false;
			pcl_u = interp3D(gd, msh, data->ustag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

			is_ugrd = false;
			is_vgrd = true;
			is_wgrd = false;
			pcl_v = interp3D(gd, msh, data->vstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

			is_ugrd = false;
			is_vgrd = false;
			is_wgrd = true;
			pcl_w = interp3D(gd, msh, data->wstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

			parcels->pclu[PCL(tidx,   parcel_id, totTime)] = pcl_u;
			parcels->pclv[PCL(tidx,   parcel_id, totTime)] = pcl_v;
			parcels->pclw[PCL(tidx,   parcel_id, totTime)] = pcl_w;

			// First RK step here. Since this is RK2,
			// I see no real reason to have loop contropl flow
			// here. I think its causing more headaches than its worth.

			// integrate X position forward by the U wind
			point[0] = pcl_x + pcl_u * dt * direct;
			// integrate Y position forward by the V wind
			point[1] = pcl_y + pcl_v * dt * direct;
			// integrate Z position forward by the W wind
			point[2] = pcl_z + pcl_w * dt * direct;

			if ((point[2] > zf(gd->NZ)) || (point[0] > xf(gd->NX)) || (point[0] > yf(gd->NY)) || \
			    (point[2] < zf(0))      || (point[0] < xf(0))      || (point[1] < yf(0))) {
				point[0] = fill;
				point[1] = fill;
				point[2] = fill;
				parcels->xpos[PCL(tidx+1, parcel_id, totTime)] = point[0]; 
				parcels->ypos[PCL(tidx+1, parcel_id, totTime)] = point[1];
				parcels->zpos[PCL(tidx+1, parcel_id, totTime)] = point[2];
				continue;
			}

			uu1 = pcl_u;
			vv1 = pcl_v;
			ww1 = pcl_w;

			is_ugrd = true;
			is_vgrd = false;
			is_wgrd = false;
			pcl_u = interp3D(gd, msh, data->ustag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

			is_ugrd = false;
			is_vgrd = true;
			is_wgrd = false;
			pcl_v = interp3D(gd, msh, data->vstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

			is_ugrd = false;
			is_vgrd = false;
			is_wgrd = true;
			pcl_w = interp3D(gd, msh, data->wstag, point, is_ugrd, is_vgrd, is_wgrd, tidx);

			// integrate X position forward by the U wind
			point[0] = pcl_x + (pcl_u + uu1) * dt2 * direct;
			// integrate Y position forward by the V wind
			point[1] = pcl_y + (pcl_v + vv1) * dt2 * direct;
			// integrate Z position forward by the W wind
			point[2] = pcl_z + (pcl_w + ww1) * dt2 * direct;

			parcels->xpos[PCL(tidx+1, parcel_id, totTime)] = point[0]; 
			parcels->ypos[PCL(tidx+1, parcel_id, totTime)] = point[1];
			parcels->zpos[PCL(tidx+1, parcel_id, totTime)] = point[2];
		} // end time loop
	} // end index check
}

__global__ void parcel_interp(grid *gd, mesh *msh, sounding *snd, parcel_pos *parcels, model_data *data, \
						  float fill_val, int tStart, int tEnd, int totTime, int direct) {

	//int parcel_id = blockIdx.x;
	int parcel_id = blockIdx.x * blockDim.x + threadIdx.x;
	// get the io config from the user namelist
	iocfg *io = parcels->io;

	// safety check to make sure our thread index doesn't
	// go out of our array bounds
	if (parcel_id < parcels->nParcels) {
		bool is_ugrd = false;
		bool is_vgrd = false;
		bool is_wgrd = false;

		float point[3];

		// loop over the number of time steps we are
		// integrating over
		for (int tidx = tStart; tidx < tEnd; ++tidx) {
			point[0] = parcels->xpos[PCL(tidx, parcel_id, totTime)];
			point[1] = parcels->ypos[PCL(tidx, parcel_id, totTime)];
			point[2] = parcels->zpos[PCL(tidx, parcel_id, totTime)];
			if (( point[0] == fill_val) || ( point[1] == fill_val ) || ( point[2] == fill_val) ) {
				if (io->output_kmh) {
					float pclkmh = fill_val; 
					parcels->pclkmh[PCL(tidx, parcel_id, totTime)] = pclkmh;
				}

				if (io->output_momentum_budget) {
					float pclupgrad = fill_val; 
					float pcluturb = fill_val; 
					float pcludiff = fill_val;
					float pclvpgrad = fill_val;
					float pclvturb = fill_val;
					float pclvdiff = fill_val;
					float pclwpgrad = fill_val;
					float pclwturb = fill_val;
					float pclwdiff = fill_val;
					float pclbuoy = fill_val;
					parcels->pclupgrad[PCL(tidx,   parcel_id, totTime)] = pclupgrad;
					parcels->pclvpgrad[PCL(tidx,   parcel_id, totTime)] = pclvpgrad;
					parcels->pclwpgrad[PCL(tidx,   parcel_id, totTime)] = pclwpgrad;
					parcels->pcluturb[PCL(tidx,   parcel_id, totTime)] = pcluturb;
					parcels->pclvturb[PCL(tidx,   parcel_id, totTime)] = pclvturb;
					parcels->pclwturb[PCL(tidx,   parcel_id, totTime)] = pclwturb;
					parcels->pcludiff[PCL(tidx,   parcel_id, totTime)] = pcludiff;
					parcels->pclvdiff[PCL(tidx,   parcel_id, totTime)] = pclvdiff;
					parcels->pclwdiff[PCL(tidx,   parcel_id, totTime)] = pclwdiff;
					parcels->pclbuoy[PCL(tidx,   parcel_id, totTime)] = pclbuoy;
				}
				if (io->output_vorticity_budget || io->output_xvort) {
					float pclxvort = fill_val;
					parcels->pclxvort[PCL(tidx, parcel_id, totTime)] = pclxvort;
				}
				if (io->output_vorticity_budget || io->output_yvort) {
					float pclyvort = fill_val;
					parcels->pclyvort[PCL(tidx, parcel_id, totTime)] = pclyvort;
				}
				if (io->output_vorticity_budget || io->output_zvort) {
					float pclzvort = fill_val;
					parcels->pclzvort[PCL(tidx, parcel_id, totTime)] = pclzvort;
				}
				if (io->output_vorticity_budget) {
					float pclxvorttilt = fill_val;
					float pclyvorttilt = fill_val;
					float pclzvorttilt = fill_val;
					float pclxvortstretch = fill_val;
					float pclyvortstretch = fill_val;
					float pclzvortstretch = fill_val;
					float pclxvortturb = fill_val;
					float pclyvortturb = fill_val;
					float pclzvortturb = fill_val;
					float pclxvortdiff = fill_val;
					float pclyvortdiff = fill_val;
					float pclzvortdiff = fill_val;
					float pclxvortbaro = fill_val;
					float pclyvortbaro = fill_val;
					float pclxvortsolenoid = fill_val;
					float pclyvortsolenoid = fill_val;
					float pclzvortsolenoid = fill_val;
					// Store the vorticity in the parcel
					parcels->pclxvorttilt[PCL(tidx, parcel_id, totTime)] = pclxvorttilt;
					parcels->pclyvorttilt[PCL(tidx, parcel_id, totTime)] = pclyvorttilt;
					parcels->pclzvorttilt[PCL(tidx, parcel_id, totTime)] = pclzvorttilt;
					parcels->pclxvortstretch[PCL(tidx, parcel_id, totTime)] = pclxvortstretch;
					parcels->pclyvortstretch[PCL(tidx, parcel_id, totTime)] = pclyvortstretch;
					parcels->pclzvortstretch[PCL(tidx, parcel_id, totTime)] = pclzvortstretch;
					parcels->pclxvortturb[PCL(tidx, parcel_id, totTime)] = pclxvortturb;
					parcels->pclyvortturb[PCL(tidx, parcel_id, totTime)] = pclyvortturb;
					parcels->pclzvortturb[PCL(tidx, parcel_id, totTime)] = pclzvortturb;
					parcels->pclxvortdiff[PCL(tidx, parcel_id, totTime)] = pclxvortdiff;
					parcels->pclyvortdiff[PCL(tidx, parcel_id, totTime)] = pclyvortdiff;
					parcels->pclzvortdiff[PCL(tidx, parcel_id, totTime)] = pclzvortdiff;
					parcels->pclxvortbaro[PCL(tidx, parcel_id, totTime)] = pclxvortbaro;
					parcels->pclyvortbaro[PCL(tidx, parcel_id, totTime)] = pclyvortbaro;
					parcels->pclxvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclxvortsolenoid;
					parcels->pclyvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclyvortsolenoid;
					parcels->pclzvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclzvortsolenoid;
				}

				// Now do the scalars
				if (io->output_ppert) {
					float pclppert = fill_val;
					parcels->pclppert[PCL(tidx, parcel_id, totTime)] = pclppert;
				}
				if (io->output_qvpert) {
					float pclqvpert = fill_val;
					parcels->pclqvpert[PCL(tidx, parcel_id, totTime)] = pclqvpert;
				}
				if (io->output_rhopert) {
					float pclrhopert = fill_val;
					parcels->pclrhopert[PCL(tidx, parcel_id, totTime)] = pclrhopert;
				}
				if (io->output_thetapert) {
					float pclthetapert = fill_val;
					parcels->pclthetapert[PCL(tidx, parcel_id, totTime)] = pclthetapert;
				}
				if (io->output_thrhopert) {
					float pclthrhopert = fill_val;
					parcels->pclthrhopert[PCL(tidx, parcel_id, totTime)] = pclthrhopert;
				}

				if (io->output_pbar) {
					float pclpbar = fill_val;
					parcels->pclpbar[PCL(tidx, parcel_id, totTime)] = pclpbar;
				}
				if (io->output_qvbar) {
					float pclqvbar = fill_val;
					parcels->pclqvbar[PCL(tidx, parcel_id, totTime)] = pclqvbar;
				}
				if (io->output_rhobar) {
					float pclrhobar = fill_val;
					parcels->pclrhobar[PCL(tidx, parcel_id, totTime)] = pclrhobar;
				}
				if (io->output_thetabar) {
					float pclthetabar = fill_val;
					parcels->pclthetabar[PCL(tidx, parcel_id, totTime)] = pclthetabar;
				}
				if (io->output_rhobar) {
					float pclthrhobar = fill_val;
					parcels->pclthrhobar[PCL(tidx, parcel_id, totTime)] = pclthrhobar;
				}

				if (io->output_qc) {
					float pclqc = fill_val;
					parcels->pclqc[PCL(tidx, parcel_id, totTime)] = pclqc;
				}
				if (io->output_qi) {
					float pclqi = fill_val;
					parcels->pclqi[PCL(tidx, parcel_id, totTime)] = pclqi;
				}
				if (io->output_qs) {
					float pclqs = fill_val;
					parcels->pclqs[PCL(tidx, parcel_id, totTime)] = pclqs;
				}
				if (io->output_qg) {
					float pclqg = fill_val; 
					parcels->pclqg[PCL(tidx, parcel_id, totTime)] = pclqg;
				}
				if (io->output_qr) {
					float pclqr = fill_val; 
					parcels->pclqr[PCL(tidx, parcel_id, totTime)] = pclqr;
				}
				continue;
			}

			if (io->output_kmh) {
				is_ugrd = false;
				is_vgrd = false;
				is_wgrd = true;
				float pclkmh = interp3D(gd, msh, data->kmh, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclkmh[PCL(tidx, parcel_id, totTime)] = pclkmh;
			}

			if (io->output_momentum_budget) {
				is_ugrd = true;
				is_vgrd = false;
				is_wgrd = false;
				float pclupgrad = interp3D(gd, msh, data->pgradu, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pcluturb = interp3D(gd, msh, data->turbu, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pcludiff = interp3D(gd, msh, data->diffu, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				is_ugrd = false;
				is_vgrd = true;
				is_wgrd = false;
				float pclvpgrad = interp3D(gd, msh, data->pgradv, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclvturb = interp3D(gd, msh, data->turbv, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclvdiff = interp3D(gd, msh, data->diffv, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				is_ugrd = false;
				is_vgrd = false;
				is_wgrd = true;
				float pclwpgrad = interp3D(gd, msh, data->pgradw, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclwturb = interp3D(gd, msh, data->turbw, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclwdiff = interp3D(gd, msh, data->diffw, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclbuoy = interp3D(gd, msh, data->buoy, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclupgrad[PCL(tidx,   parcel_id, totTime)] = pclupgrad;
				parcels->pclvpgrad[PCL(tidx,   parcel_id, totTime)] = pclvpgrad;
				parcels->pclwpgrad[PCL(tidx,   parcel_id, totTime)] = pclwpgrad;
				parcels->pcluturb[PCL(tidx,   parcel_id, totTime)] = pcluturb;
				parcels->pclvturb[PCL(tidx,   parcel_id, totTime)] = pclvturb;
				parcels->pclwturb[PCL(tidx,   parcel_id, totTime)] = pclwturb;
				parcels->pcludiff[PCL(tidx,   parcel_id, totTime)] = pcludiff;
				parcels->pclvdiff[PCL(tidx,   parcel_id, totTime)] = pclvdiff;
				parcels->pclwdiff[PCL(tidx,   parcel_id, totTime)] = pclwdiff;
				parcels->pclbuoy[PCL(tidx,   parcel_id, totTime)] = pclbuoy;
			}


			// interpolate scalar values to the parcel point
			is_ugrd = false;
			is_vgrd = false;
			is_wgrd = false;
			if (io->output_vorticity_budget || io->output_xvort) {
				float pclxvort = interp3D(gd, msh, data->xvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclxvort[PCL(tidx, parcel_id, totTime)] = pclxvort;
			}
			if (io->output_vorticity_budget || io->output_yvort) {
				float pclyvort = interp3D(gd, msh, data->yvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclyvort[PCL(tidx, parcel_id, totTime)] = pclyvort;
			}
			if (io->output_vorticity_budget || io->output_zvort) {
				float pclzvort = interp3D(gd, msh, data->zvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclzvort[PCL(tidx, parcel_id, totTime)] = pclzvort;
			}
			if (io->output_vorticity_budget) {
				float pclxvorttilt = interp3D(gd, msh, data->xvtilt, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclyvorttilt = interp3D(gd, msh, data->yvtilt, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclzvorttilt = interp3D(gd, msh, data->zvtilt, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclxvortstretch = interp3D(gd, msh, data->xvstretch, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclyvortstretch = interp3D(gd, msh, data->yvstretch, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclzvortstretch = interp3D(gd, msh, data->zvstretch, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclxvortturb = interp3D(gd, msh, data->turbxvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclyvortturb = interp3D(gd, msh, data->turbyvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclzvortturb = interp3D(gd, msh, data->turbzvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclxvortdiff = interp3D(gd, msh, data->diffxvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclyvortdiff = interp3D(gd, msh, data->diffyvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclzvortdiff = interp3D(gd, msh, data->diffzvort, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclxvortbaro = interp3D(gd, msh, data->xvort_baro, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclyvortbaro = interp3D(gd, msh, data->yvort_baro, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclxvortsolenoid = interp3D(gd, msh, data->xvort_solenoid, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclyvortsolenoid = interp3D(gd, msh, data->yvort_solenoid, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				float pclzvortsolenoid = interp3D(gd, msh, data->zvort_solenoid, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				// Store the vorticity in the parcel
				parcels->pclxvorttilt[PCL(tidx, parcel_id, totTime)] = pclxvorttilt;
				parcels->pclyvorttilt[PCL(tidx, parcel_id, totTime)] = pclyvorttilt;
				parcels->pclzvorttilt[PCL(tidx, parcel_id, totTime)] = pclzvorttilt;
				parcels->pclxvortstretch[PCL(tidx, parcel_id, totTime)] = pclxvortstretch;
				parcels->pclyvortstretch[PCL(tidx, parcel_id, totTime)] = pclyvortstretch;
				parcels->pclzvortstretch[PCL(tidx, parcel_id, totTime)] = pclzvortstretch;
				parcels->pclxvortturb[PCL(tidx, parcel_id, totTime)] = pclxvortturb;
				parcels->pclyvortturb[PCL(tidx, parcel_id, totTime)] = pclyvortturb;
				parcels->pclzvortturb[PCL(tidx, parcel_id, totTime)] = pclzvortturb;
				parcels->pclxvortdiff[PCL(tidx, parcel_id, totTime)] = pclxvortdiff;
				parcels->pclyvortdiff[PCL(tidx, parcel_id, totTime)] = pclyvortdiff;
				parcels->pclzvortdiff[PCL(tidx, parcel_id, totTime)] = pclzvortdiff;
				parcels->pclxvortbaro[PCL(tidx, parcel_id, totTime)] = pclxvortbaro;
				parcels->pclyvortbaro[PCL(tidx, parcel_id, totTime)] = pclyvortbaro;
				parcels->pclxvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclxvortsolenoid;
				parcels->pclyvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclyvortsolenoid;
				parcels->pclzvortsolenoid[PCL(tidx, parcel_id, totTime)] = pclzvortsolenoid;
			}

			// Now do the scalars
			if (io->output_ppert) {
				float pclppert = interp3D(gd, msh, data->prespert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclppert[PCL(tidx, parcel_id, totTime)] = pclppert;
			}
			if (io->output_qvpert) {
				float pclqvpert = interp3D(gd, msh, data->qvpert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclqvpert[PCL(tidx, parcel_id, totTime)] = pclqvpert;
			}
			if (io->output_rhopert) {
				float pclrhopert = interp3D(gd, msh, data->rhopert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclrhopert[PCL(tidx, parcel_id, totTime)] = pclrhopert;
			}
			if (io->output_thetapert) {
				float pclthetapert = interp3D(gd, msh, data->thetapert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclthetapert[PCL(tidx, parcel_id, totTime)] = pclthetapert;
			}
			if (io->output_thrhopert) {
				float pclthrhopert = interp3D(gd, msh, data->thrhopert, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclthrhopert[PCL(tidx, parcel_id, totTime)] = pclthrhopert;
			}

			if (io->output_pbar) {
				float pclpbar = interp1D(gd, msh, snd->pres0, point[2], is_wgrd, tidx);
				parcels->pclpbar[PCL(tidx, parcel_id, totTime)] = pclpbar;
			}
			if (io->output_qvbar) {
				float pclqvbar = interp1D(gd, msh, snd->qv0, point[2], is_wgrd, tidx);
				parcels->pclqvbar[PCL(tidx, parcel_id, totTime)] = pclqvbar;
			}
			if (io->output_rhobar) {
				float pclrhobar = interp1D(gd, msh, snd->rho0, point[2], is_wgrd, tidx);
				parcels->pclrhobar[PCL(tidx, parcel_id, totTime)] = pclrhobar;
			}
			if (io->output_thetabar) {
				float pclthetabar = interp1D(gd, msh, snd->th0, point[2], is_wgrd, tidx);
				parcels->pclthetabar[PCL(tidx, parcel_id, totTime)] = pclthetabar;
			}
			if (io->output_rhobar) {
				float qvbar = interp1D(gd, msh, snd->qv0, point[2], is_wgrd, tidx);
				float pclthrhobar = interp1D(gd, msh, snd->th0, point[2], is_wgrd, tidx);
				pclthrhobar= pclthrobar*(1.0+reps*qvbar)/(1.0+qvbar);
				parcels->pclthrhobar[PCL(tidx, parcel_id, totTime)] = pclthrhobar;
			}

			if (io->output_qc) {
				float pclqc = interp3D(gd, msh, data->qc, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclqc[PCL(tidx, parcel_id, totTime)] = pclqc;
			}
			if (io->output_qi) {
				float pclqi = interp3D(gd, msh, data->qi, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclqi[PCL(tidx, parcel_id, totTime)] = pclqi;
			}
			if (io->output_qs) {
				float pclqs = interp3D(gd, msh, data->qs, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclqs[PCL(tidx, parcel_id, totTime)] = pclqs;
			}
			if (io->output_qg) {
				float pclqg = interp3D(gd, msh, data->qg, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclqg[PCL(tidx, parcel_id, totTime)] = pclqg;
			}
			if (io->output_qr) {
				float pclqr = interp3D(gd, msh, data->qr, point, is_ugrd, is_vgrd, is_wgrd, tidx);
				parcels->pclqr[PCL(tidx, parcel_id, totTime)] = pclqr;
			}
		}
	}
}

/* This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff */
void cudaIntegrateParcels(grid *gd_cpu, mesh *msh, sounding *snd,  model_data *data, parcel_pos *parcels, float fill, int nT, int totTime, int direct) {
	int tStart, tEnd;
	tStart = 0;
	tEnd = nT;
	int NX, NY, NZ;

	// we need to create a grid struct
	// on the GPU
	grid *gd;
    cudaMallocManaged(&gd, sizeof(grid));
	gd->X0 = gd_cpu->X0; gd->X1 = gd_cpu->X1;
	gd->Y0 = gd_cpu->Y0; gd->Y1 = gd_cpu->Y1;
	gd->Z0 = gd_cpu->Z0; gd->Z1 = gd_cpu->Z1;
	// BEFORE WE INTEGRATE THE PARCELS!!!
	// Up to this point, our array buffers of data (eg U, V, W) 
	// and our mesh dimensions are the same size. In the calculation
	// routines, we use special macros that make working with the staggered
	// data a little more convenient. We are goind to reduce NX/NY/NZ appripriately
	// such that these macros don't access outside of memory bounds.
	gd->NX = gd_cpu->NX-2; gd->NY = gd_cpu->NY-2; gd->NZ = gd_cpu->NZ-1;

	NX = gd->NX;
	NY = gd->NY;
	NZ = gd->NZ;
	iocfg *io = parcels->io;

	cudaStream_t memStream;
	cudaStreamCreate(&memStream);
	long bufsize = (gd_cpu->NX)*(gd_cpu->NY)*(gd_cpu->NZ)*nT;
	prefetch_model_gpu(io, data, bufsize, memStream);

	cudaStream_t intStream;
	cudaStreamCreate(&intStream);
	cout << endl << endl;


	// set the thread/block execution strategy for the kernels
	dim3 threadsPerBlock(256, 1, 1);
	dim3 numBlocks((int)ceil(NX+2/threadsPerBlock.x)+1, (int)ceil(NY+2/threadsPerBlock.y)+1, (int)ceil(NZ+1/threadsPerBlock.z)+1); 

	auto totstart = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	if (io->output_momentum_budget) {
		start = std::chrono::high_resolution_clock::now();
		doMomentumBud(gd, msh, snd, data, tStart, tEnd, numBlocks, threadsPerBlock);
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		cout << "Momentum Budget Calculation Time: " << duration.count() << " ms" << endl;
	}
	// Calculate the three compionents of vorticity
	// and do the necessary averaging. This is a wrapper that
	// calls the necessary kernels and assigns the pointers
	// appropriately such that the "user" only has to call this method.
	if (io->output_xvort || io->output_yvort || io->output_zvort || io->output_vorticity_budget) {
		start = std::chrono::high_resolution_clock::now();
		doCalcVort(gd, msh, data, tStart, tEnd, numBlocks, threadsPerBlock);
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		cout << "Vorticity Calculation Time: " << duration.count() << " ms" << endl;
	}
	

	// Calculate the vorticity forcing terms for each of the 3 components.
	// This is a wrapper that calls the necessary kernels to compute the
	// derivatives and average them back to the scalar grid where necessary. 
	if (io->output_vorticity_budget) {
		start = std::chrono::high_resolution_clock::now();
		doCalcVortTend(gd, msh, snd, data, tStart, tEnd, numBlocks, threadsPerBlock);
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		cout << "Vorticity Budget Calculation Time: " << duration.count() << " ms" << endl;
	}


	// Before integrating the trajectories, George Bryan sets some below-grid/surface conditions 
	// that we need to consider. This handles applying those boundary conditions. 
	//applyMomentumBC<<<numBlocks, threadsPerBlock>>>(data->ustag, data->vstag, data->wstag, NX, NY, NZ, tStart, tEnd);
	//gpuErrchk(cudaDeviceSynchronize() );
	//gpuErrchk( cudaPeekAtLastError() );

	// integrate the parcels forward in time and interpolate
	// calculations to trajectories. 
	int nThreads = 256;
	int nPclBlocks = int(parcels->nParcels / nThreads) + 1;
	start = std::chrono::high_resolution_clock::now();
	integrate<<<nPclBlocks, nThreads, 0, intStream>>>(gd, msh, snd, parcels, data, fill, tStart, tEnd, totTime, direct);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk( cudaPeekAtLastError() );

	parcel_interp<<<nPclBlocks, nThreads, 0, intStream>>>(gd, msh, snd, parcels, data, fill, tStart, tEnd, totTime, direct);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk( cudaPeekAtLastError() );
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	cout << "Parcel Integration and Interoplation Time: " << duration.count() << " ms" << endl;
	auto totend = std::chrono::high_resolution_clock::now();
	auto totdur = std::chrono::duration_cast<std::chrono::milliseconds>(totend - totstart);
	cout << "Total time spent on GPU: " << totdur.count() << " ms" << endl;
	cout << endl << endl;

	gpuErrchk( cudaFree(gd) );

	prefetch_parcels_cpu(io, parcels, memStream);
	cudaStreamDestroy(memStream);
}
#endif

