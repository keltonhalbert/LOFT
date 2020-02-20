#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#include "../calc/calcvort.cu"
#ifndef VORT_CU
#define VORT_CU

__global__ void calcpipert(datagrid *grid, float *prespert, float *pipert) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    if ((i < NX+1) && (j < NY+1) && (k < NZ)) {
		calc_pipert(prespert, grid->p0, pipert, i, j, k, NX, NY);
    }
}

__global__ void calcxvort(datagrid *grid, float *vstag, float *wstag, float *xvort) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float dy, dz;
	float *buf0 = xvort;

    if ((i < NX) && (j < NY+1) && (k > 0) && (k < NZ)) {
        dy = yf(j) - yf(j-1);
        dz = zf(k) - zf(k-1);

		calc_xvort(vstag, wstag, xvort, dy, dz, i, j, k, NX, NY);
		// lower boundary condition of stencil
		if ((k == 1) && (zf(k-1) == 0)) {
			BUF(i, j, 0) = BUF(i, j, 1);
		}
    }
}

__global__ void calcyvort(datagrid *grid, float *ustag, float *wstag, float *yvort) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float dx, dz;
	float *buf0 = yvort;

    if ((i < NX+1) && (j < NY) && (k > 0) && (k < NZ+1)) {
        dx = xf(i) - xf(i-1);
        dz = zf(k) - zf(k-1);

		calc_yvort(ustag, wstag, yvort, dx, dz, i, j, k, NX, NY);
		// lower boundary condition of stencil
		if ((k == 1) && (zf(k-1) == 0)) {
			BUF(i, j, 0) = BUF(i, j, 1);
		}
    }
}
__global__ void calczvort(datagrid *grid, float *ustag, float *vstag, float *zvort) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float dx, dy;

    if ((i < NX+1) && (j < NY+1) && (k < NZ+1)) {
        dx = xf(i) - xf(i-1);
        dy = yf(j) - yf(j-1);

		calc_zvort(ustag, vstag, zvort, dx, dy, i, j, k, NX, NY);
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calcvortstretch(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    long bufidx;
    float dx, dy, dz;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        dy = yf(j+1) - yf(j);
        dz = zf(k+1) - zf(k);
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
            calc_xvort_stretch(&(data->vstag[bufidx]), &(data->wstag[bufidx]), \
                               &(data->xvort[bufidx]), &(data->xvstretch[bufidx]), \
                               dy, dz, i, j, k, NX, NY);
            if ((k == 1) && (zf(k-1) == 0)) {
                data->xvstretch[P4(i, j, 0, tidx, NX+2, NY+2, NZ+1)] = data->xvstretch[P4(i, j, 1, tidx, NX+2, NY+2, NZ+1)];
            }
        }
    }

    if ((i < NX) && (j < NY) && (k < NZ)) {
        dx = xf(i+1) - xf(i);
        dz = zf(k+1) - zf(k);
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
            calc_yvort_stretch(&(data->ustag[bufidx]), &(data->wstag[bufidx]), \
                               &(data->yvort[bufidx]), &(data->yvstretch[bufidx]), \
                               dx, dz, i, j, k, NX, NY);
            if ((k == 1) && (zf(k-1) == 0)) {
                data->yvstretch[P4(i, j, 0, tidx, NX+2, NY+2, NZ+1)] = data->yvstretch[P4(i, j, 1, tidx, NX+2, NY+2, NZ+1)];
            }
        }
    }

    if ((i < NX) && (j < NY) && (k < NZ)) {
        dx = xf(i+1) - xf(i);
        dz = yf(j+1) - yf(j);
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
            calc_zvort_stretch(&(data->ustag[bufidx]), &(data->vstag[bufidx]), \
                               &(data->zvort[bufidx]), &(data->zvstretch[bufidx]), \
                               dx, dy, i, j, k, NX, NY);
        }
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calcxvorttilt(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    //printf("%i, %i, %i\n", i, j, k);

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort_tilt(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calcyvorttilt(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    //printf("%i, %i, %i\n", i, j, k);

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_yvort_tilt(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calczvorttilt(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    //printf("%i, %i, %i\n", i, j, k);

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvort_tilt(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the buoyancy/baroclinic term */ 
__global__ void calcvortbaro(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    long bufidx;
    float dx, dy;

    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k > 0)) {
        // loop over the number of time steps we have in memory
        dx = xh(i+1) - xh(i-1);
        dy = yh(j+1) - yh(j-1);
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
            calc_xvort_baro(&(data->thrhopert[bufidx]), grid->th0, grid->qv0, &(data->xvort_baro[bufidx]), dx, i, j, k, NX, NY);
            calc_yvort_baro(&(data->thrhopert[bufidx]), grid->th0, grid->qv0, &(data->yvort_baro[bufidx]), dy, i, j, k, NX, NY);
        }
    }
}

/* Compute the forcing tendencies from the pressure-volume solenoid term */
__global__ void calcvortsolenoid(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float dx, dy, dz;
    long bufidx;

    // Even though there are NZ points, it's a center difference
    // and we reach out NZ+1 points to get the derivatives
    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0)) {
        dx = xh(i+1)-xh(i-1);
        dy = yh(j+1)-yh(j-1);
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
            calc_zvort_solenoid(&(data->pipert[bufidx]), &(data->thrhopert[bufidx]), \
                                &(data->zvort_solenoid[bufidx]), dx, dy, i, j, k, NX, NY);
        }
    }
    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k > 0)) {
        // loop over the number of time steps we have in memory
        dx = xh(i+1)-xh(i-1);
        dy = yh(j+1)-yh(j-1);
        dz = zh(k+1)-zh(k-1);
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            bufidx = P4(0, 0, 0, tidx, NX+2, NY+2, NZ+1);
            calc_xvort_solenoid(&(data->pipert[bufidx]), &(data->thrhopert[bufidx]), grid->th0, grid->qv0, \
                                &(data->xvort_solenoid[bufidx]), dy, dz, i, j, k, NX, NY);
            calc_yvort_solenoid(&(data->pipert[bufidx]), &(data->thrhopert[bufidx]), grid->th0, grid->qv0, \
                                &(data->yvort_solenoid[bufidx]), dx, dz, i, j, k, NX, NY);
            if ((k == 1) && (zf(k-1) == 0)) {
                data->xvort_solenoid[P4(i, j, 0, tidx, NX+2, NY+2, NZ+1)] = data->xvort_solenoid[P4(i, j, 1, tidx, NX+2, NY+2, NZ+1)];
                data->yvort_solenoid[P4(i, j, 0, tidx, NX+2, NY+2, NZ+1)] = data->yvort_solenoid[P4(i, j, 1, tidx, NX+2, NY+2, NZ+1)];
            }
        }
    }
}

/* Zero out the temporary arrays */
__global__ void zeroTemArrays(datagrid *grid, model_data *data, int tStart, int tEnd) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *dum0;
    if (( i < NX+1) && ( j < NY+1) && ( k < NZ+1)) {
        dum0 = data->tem1;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem2;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem3;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem4;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem5;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem6;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
    }
}


/* Average our vorticity values back to the scalar grid for interpolation
   to the parcel paths. We're able to do this in parallel by making use of
   the three temporary arrays allocated on our grid, which means that the
   xvort/yvort/zvort arrays will be averaged into tem1/tem2/tem3. After
   calling this kernel, you MUST set the new pointers appropriately. */
__global__ void doVortAvg(datagrid *grid, model_data *data, int tStart, int tEnd) {

    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // average the temporary arrays into the result arrays
            dum0 = data->tem1;
            buf0 = data->xvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem2;
            buf0 = data->yvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem3;
            buf0 = data->zvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );
        }
    }
}

__global__ void doTurbVortAvg(datagrid *grid, model_data *data, int tStart, int tEnd) {

    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // average the temporary arrays into the result arrays
            dum0 = data->tem1;
            buf0 = data->turbxvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem2;
            buf0 = data->turbyvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem3;
            buf0 = data->turbzvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );
        }
    }
}


__global__ void doDiffVortAvg(datagrid *grid, model_data *data, int tStart, int tEnd) {

    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // average the temporary arrays into the result arrays
            dum0 = data->tem1;
            buf0 = data->diffxvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem2;
            buf0 = data->diffyvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem3;
            buf0 = data->diffzvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );
        }
    }
}

/* Average the derivatives within the temporary arrays used to compute
   the tilting rate and then combine the terms into the final xvtilt
   array. It is assumed that the derivatives have been precomputed into
   the temporary arrays. */
__global__ void doXVortTiltAvg(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;
    float dudy,dudz;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1;
            //dudy = TEM4D(i, j, k, tidx);
            dudy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            dum0 = data->tem2;
            //dudz = TEM4D(i, j, k, tidx);
            dudz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            buf0 = data->zvort;
            float zvort = BUF4D(i, j, k, tidx);
            buf0 = data->yvort;
            float yvort = BUF4D(i, j, k, tidx);

            buf0 = data->xvtilt;
            BUF4D(i, j, k, tidx) = zvort * dudz + yvort * dudy; 
        }
    }
}

/* Average the derivatives within the temporary arrays used to compute
   the tilting rate and then combine the terms into the final yvtilt
   array. It is assumed that the derivatives have been precomputed into
   the temporary arrays. */
__global__ void doYVortTiltAvg(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;
    float dvdx, dvdz;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1;
            //dvdx = TEM4D(i, j, k, tidx);
            dvdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            dum0 = data->tem2;
            //dvdz = TEM4D(i, j, k, tidx);
            dvdz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            buf0 = data->xvort;
            float xvort = BUF4D(i, j, k, tidx);
            buf0 = data->zvort;
            float zvort = BUF4D(i, j, k, tidx);

            buf0 = data->yvtilt;
            BUF4D(i, j, k, tidx) = xvort * dvdx + zvort * dvdz; 
        }
    }
}

/* Average the derivatives within the temporary arrays used to compute
   the tilting rate and then combine the terms into the final zvtilt
   array. It is assumed that the derivatives have been precomputed into
   the temporary arrays. */
__global__ void doZVortTiltAvg(datagrid *grid, model_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;
    float dwdx, dwdy;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1;
            //dwdx = TEM4D(i, j, k, tidx);
            dwdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem2;
            dwdy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );
            //dwdy = TEM4D(i, j, k, tidx);
            buf0 = data->xvort;
            float xvort = BUF4D(i, j, k, tidx);
            buf0 = data->yvort;
            float yvort = BUF4D(i, j, k, tidx);
            
            buf0 = data->zvtilt;
            BUF4D(i, j, k, tidx) = xvort * dwdx + yvort * dwdy; 
        }
    }
}

#endif
