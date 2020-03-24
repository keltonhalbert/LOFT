#include <iostream>
#include <stdio.h>
extern "C" {
#include <lofs-read.h>
#include <dirstruct.h>
#include <hdf2nc.h>
#include <limits.h>
#include <macros.h>
}
#include "../include/datastructs.h"
#include "../calc/calcvort.cu"
#ifndef VORT_CU
#define VORT_CU

/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/
__global__ void cuCalcPipert(grid *gd, sounding *snd, float *prespert, float *pipert) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    if ((i < NX+1) && (j < NY+1) && (k < NZ)) {
		calc_pipert(prespert, snd->p0, pipert, i, j, k, NX, NY);
    }
}

__global__ void cuCalcXvort(grid *gd, mesh *msh, float *vstag, float *wstag, float *xvort) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
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

__global__ void cuCalcYvort(grid *gd, mesh *msh, float *ustag, float *wstag, float *yvort) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
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

__global__ void cuCalcZvort(grid *gd, mesh *msh, float *ustag, float *vstag, float *zvort) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dx, dy;

    if ((i < NX+1) && (j < NY+1) && (k < NZ+1)) {
        dx = xf(i) - xf(i-1);
        dy = yf(j) - yf(j-1);

		calc_zvort(ustag, vstag, zvort, dx, dy, i, j, k, NX, NY);
    }
}

__global__ void cuCalcXvortStretch(grid *gd, mesh *msh, float *vstag, float *wstag, float *xvort, float *xvstretch) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dy, dz;


    if ((i < NX) && (j < NY) && (k < NZ)) {
        dy = yf(j+1) - yf(j);
        dz = zf(k+1) - zf(k);
		    calc_xvort_stretch(vstag, wstag, xvort, xvstretch, dy, dz, i, j, k, NX, NY);
    }
}

__global__ void cuCalcYvortStretch(grid *gd, mesh *msh, float *ustag, float *wstag, float *yvort, float *yvstretch) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dx, dz;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        dx = xf(i+1) - xf(i);
        dz = zf(k+1) - zf(k);
		calc_yvort_stretch(ustag, wstag, yvort, yvstretch, dx, dz, i, j, k, NX, NY);
    }

}
/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void cuCalcZvortStretch(grid *gd, mesh *msh, float *ustag, float *vstag, float *zvort, float *zvstretch) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dx, dy;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        dx = xf(i+1) - xf(i);
        dy = yf(j+1) - yf(j);
		calc_zvort_stretch(ustag, vstag, zvort, zvstretch, dx, dy, i, j, k, NX, NY);
    }
}

__global__ void cuPreXvortTilt(grid *gd, mesh *msh, sounding *snd, float *ustag, float *dudy, float *dudz) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
	float dy, dz;

    if ((i < NX) && (j < NY) && (k < NZ)) {
		dy = yf(j) - yf(j-1);
        // loop over the number of time steps we have in memory
		calc_dudy(ustag, dudy, dy, i, j, k, NX, NY);
    }

	if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
		dz = zf(k) - zf(k-1);
		calc_dudz(ustag, dudz, dz, i, j, k, NX, NY);
	}
}

__global__ void cuPreYvortTilt(grid *gd, mesh *msh, float *vstag, float *dvdx, float *dvdz) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
	float dx, dz;

    if ((i < NX) && (j < NY) && (k < NZ)) {
		dx = xf(i) - xf(i-1);
        // loop over the number of time steps we have in memory
		calc_dvdx(vstag, dvdx, dx, i, j, k, NX, NY);
    }

	if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
		dz = zf(k) - zf(k-1);
		calc_dvdz(vstag, dvdz, dz, i, j, k, NX, NY);
	}
}

__global__ void cuPreZvortTilt(grid *gd, mesh *msh, float *wstag, float *dwdx, float *dwdy) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
	float dx, dy;

    if ((i < NX) && (j < NY) && (k < NZ)) {
		dx = xf(i) - xf(i-1);
		dy = yf(j) - yf(j-1);
        // loop over the number of time steps we have in memory
		calc_dwdx(wstag, dwdx, dx, i, j, k, NX, NY);
		calc_dwdy(wstag, dwdy, dy, i, j, k, NX, NY);
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void cuCalcXvortTilt(grid *gd, mesh *msh, float *yvort, float *zvort, float *dudy, float *dudz, float *xvtilt) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        // loop over the number of time steps we have in memory
		calc_xvort_tilt(yvort, zvort, dudy, dudz, xvtilt, i, j, k, NX, NY);
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void cuCalcYvortTilt(grid *gd, mesh *msh, float *xvort, float *zvort, float *dvdx, float *dvdz, float *yvtilt) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        // loop over the number of time steps we have in memory
		calc_yvort_tilt(xvort, zvort, dvdx, dvdz, yvtilt, i, j, k, NX, NY);
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void cuCalcZvortTilt(grid *gd, mesh *msh, float *xvort, float *yvort, float *dwdx, float *dwdy, float *zvtilt) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        // loop over the number of time steps we have in memory
		calc_zvort_tilt(xvort, yvort, dwdx, dwdy, zvtilt, i, j, k, NX, NY);
    }
}

__global__ void cuCalcXvortBaro(grid *gd, mesh *msh, sounding *snd, float *thrhopert, float *xvort_baro) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dx;

    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k > 0)) {
        // loop over the number of time steps we have in memory
        dx = xh(i+1) - xh(i-1);
		calc_xvort_baro(thrhopert, snd->th0, snd->qv0, xvort_baro, dx, i, j, k, NX, NY);
    }
}

__global__ void cuCalcYvortBaro(grid *gd, mesh *msh, sounding *snd, float *thrhopert, float *yvort_baro) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dy;

    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k > 0)) {
        // loop over the number of time steps we have in memory
        dy = yh(j+1) - yh(j-1);
		calc_yvort_baro(thrhopert, snd->th0, snd->qv0, yvort_baro, dy, i, j, k, NX, NY);
    }
}

/* Compute the forcing tendencies from the pressure-volume solenoid term */
__global__ void cuCalcXvortSolenoid(grid *gd, mesh *msh, sounding *snd, float *pipert, float *thrhopert, float *xvort_solenoid) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dy, dz;

    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k > 0)) {
        dy = yh(j+1)-yh(j-1);
        dz = zh(k+1)-zh(k-1);

		calc_xvort_solenoid(pipert, thrhopert, snd->th0, snd->qv0, xvort_solenoid, dy, dz, i, j, k, NX, NY);
		if ((k == 1) && (zf(k-1) == 0)) {
			xvort_solenoid[P3(i, j, 0, NX+2, NY+2)] = xvort_solenoid[P3(i, j, 1, NX+2, NY+2)];
		}
    }
}

/* Compute the forcing tendencies from the pressure-volume solenoid term */
__global__ void cuCalcYvortSolenoid(grid *gd, mesh *msh, sounding *snd, float *pipert, float *thrhopert, float *yvort_solenoid) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dx, dz;

    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k > 0)) {
        dx = xh(i+1)-xh(i-1);
        dz = zh(k+1)-zh(k-1);

		calc_yvort_solenoid(pipert, thrhopert, snd->th0, snd->qv0, yvort_solenoid, dx, dz, i, j, k, NX, NY);
		if ((k == 1) && (zf(k-1) == 0)) {
			yvort_solenoid[P3(i, j, 0, NX+2, NY+2)] = yvort_solenoid[P3(i, j, 1, NX+2, NY+2)];
		}
    }
}

/* Compute the forcing tendencies from the pressure-volume solenoid term */
__global__ void cuCalcZvortSolenoid(grid *gd, mesh *msh, float *pipert, float *thrhopert, float *zvort_solenoid) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float dx, dy;

    // Even though there are NZ points, it's a center difference
    // and we reach out NZ+1 points to get the derivatives
    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0)) {
        dx = xh(i+1)-xh(i-1);
        dy = yh(j+1)-yh(j-1);

		calc_zvort_solenoid(pipert, thrhopert, zvort_solenoid, dx, dy, i, j, k, NX, NY);
    }
}

/* Zero out the temporary arrays */
__global__ void zeroTemArrays(grid *gd, model_data *data, int tStart, int tEnd) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
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
   xvort/yvort/zvort arrays will be averaged into tem1/tem2/tem3. */ 
__global__ void doVortAvg(grid *gd, float *tem1, float *tem2, float *tem3, float *xvort, float *yvort, float *zvort, int tStart, int tEnd) {

    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = gd->NX;
    int NY = gd->NY;
    int NZ = gd->NZ;
    float *buf0, *dum0;

    if ((i < NX) && (j < NY) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // average the temporary arrays into the result arrays
            dum0 = tem1;
            buf0 = xvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = tem2;
            buf0 = yvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = tem3;
            buf0 = zvort;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );
        }
    }
}


#endif
