#include <iostream>
#include <stdio.h>
#include "datastructs.cu"
#include "macros.cpp"
#include "interp.cu"
#ifndef VORT_CU
#define VORT_CU

/* Compute the Exner function / nondimensionalized pressure */
__device__ void calc_pi(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // This is p'
    float *buf0 = data->pres_4d_chunk;
    float pi = powf( BUF4D(i, j, k, t) / 1000., 0.28571426);
    buf0 = data->pi_4d_chunk;
    BUF4D(i, j, k, t) = pi;
}

/* Compute the x component of vorticity. After this is called by the calvort kernel, you must also run 
   the kernel for applying the lower boundary condition and then the kernel for averaging to the
   scalar grid. */
__device__ void calc_xvort(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->v_4d_chunk;
    float *wstag = data->w_4d_chunk;
    float *dum0 = data->tem1_4d_chunk;

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/grid->dy ) * VF(j);
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/grid->dz ) * MF(k);
    TEM4D(i, j, k, t) = dwdy - dvdz; 
}

/* Compute the y component of vorticity. After this is called by the calvort kernel, you must also run 
   the kernel for applying the lower boundary condition and then the kernel for averaging to the
   scalar grid. */
__device__ void calc_yvort(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->u_4d_chunk;
    float *wstag = data->w_4d_chunk;
    float *dum0 = data->tem2_4d_chunk;

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/grid->dx ) * UF(i);
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/grid->dz ) * MF(k);
    TEM4D(i, j, k, t) = dudz - dwdx;
}

/* Compute the z component of vorticity. After this is called by the calvort kernel, you must also run 
   the kernel for applying the lower boundary condition and then the kernel for averaging to the
   scalar grid. */
__device__ void calc_zvort(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->u_4d_chunk;
    float *vstag = data->v_4d_chunk;
    float *dum0 = data->tem3_4d_chunk;

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/grid->dx) * UF(i);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/grid->dy) * VF(j);
    TEM4D(i, j, k, t) = dvdx - dudy;
}

/* Compute the X component of vorticity tendency due
   to tilting Y and Z components into the X direction */
__device__ void calc_xvort_tilt(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->u_4d_chunk;
    float *vstag = data->v_4d_chunk;
    float *wstag = data->w_4d_chunk;

    // dudy in tem1
    float *dum0 = data->tem1_4d_chunk;
    TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) ) / grid->dy ) * VF(j);

    // dwdx in tem2
    dum0 = data->tem2_4d_chunk;
    TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) ) / grid->dx ) * UF(i);

    // dudz in tem3
    dum0 = data->tem3_4d_chunk;
    TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) ) / grid->dz ) * MF(k);

    // dvdx in tem4
    dum0 = data->tem4_4d_chunk;
    TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) ) / grid->dx ) * UF(i);
}

/* Compute the Y component of vorticity tendency due
   to tilting X and Z components into the X direction */
__device__ void calc_yvort_tilt(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->u_4d_chunk;
    float *vstag = data->v_4d_chunk;
    float *wstag = data->w_4d_chunk;
    
    // dvdx in tem1
    float *dum0 = data->tem1_4d_chunk;
    TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) ) / grid->dx ) * UF(i);

    // dwdy in tem2
    dum0 = data->tem2_4d_chunk;
    TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) ) / grid->dy ) * VF(j);

    // dvdz in tem3
    dum0 = data->tem3_4d_chunk;
    TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) ) / grid->dz ) * MF(k);

    // dudy in tem4
    dum0 = data->tem4_4d_chunk;
    TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) ) / grid->dy ) * VF(j);
}

/* Compute the Z component of vorticity tendency due
   to tilting X and Y components into the X direction */
__device__ void calc_zvort_tilt(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->u_4d_chunk;
    float *vstag = data->v_4d_chunk;
    float *wstag = data->w_4d_chunk;

    // Compute dw/dx and put it in the tem1 array. The derivatives
    // land on weird places so we have to average each derivative back
    // to the scalar grid, resulting in this clunky approach
    float *dum0 = data->tem1_4d_chunk;
    TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) ) / grid->dx ) * UF(i);

    // put dv/dz in tem2
    dum0 = data->tem2_4d_chunk;
    TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) ) / grid->dz ) * MF(k);

    // put dw/dy in tem3
    dum0 = data->tem3_4d_chunk;
    TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) ) / grid->dy ) * VF(j);

    // put du/dz in tem4
    dum0 = data->tem4_4d_chunk;
    TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) ) / grid->dz ) * MF(k);
}

/* Compute the X component of vorticity tendency due
   to stretching of the vorticity along the X axis. */
__device__ void calc_xvort_stretch(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->v_4d_chunk;
    float *wstag = data->w_4d_chunk;
    float *xvort = data->xvort_4d_chunk;
    float *xvort_stretch = data->xvstretch_4d_chunk;

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = xvort;
    float xv = BUF4D(i, j, k, t);
    float dvdy = ( ( VA4D(i, j, k, t) - VA4D(i, j-1, k, t) )/grid->dy) * VF(j);
    float dwdz = ( ( WA4D(i, j, k, t) - WA4D(i, j, k-1, t) )/grid->dz) * MF(k);

    buf0 = xvort_stretch;
    BUF4D(i, j, k, t) = -1.0*xv*( dvdy + dwdz);

}

/* Compute the Y component of vorticity tendency due
   to stretching of the vorticity along the Y axis. */
__device__ void calc_yvort_stretch(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->u_4d_chunk;
    float *wstag = data->w_4d_chunk;
    float *yvort = data->yvort_4d_chunk;
    float *yvort_stretch = data->yvstretch_4d_chunk;

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = yvort;
    float yv = BUF4D(i, j, k, t);
    float dudx = ( ( UA4D(i, j, k, t) - UA4D(i-1, j, k, t) )/grid->dx) * UF(i);
    float dwdz = ( ( WA4D(i, j, k, t) - WA4D(i, j, k-1, t) )/grid->dz) * MF(k);

    buf0 = yvort_stretch;
    BUF4D(i, j, k, t) = -1.0*yv*( dudx + dwdz);
}

/* Compute the Z component of vorticity tendency due
   to stretching of the vorticity along the Z axis. */
__device__ void calc_zvort_stretch(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->u_4d_chunk;
    float *vstag = data->v_4d_chunk;
    float *zvort = data->zvort_4d_chunk;
    float *zvort_stretch = data->zvstretch_4d_chunk;

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = zvort;
    float zv = BUF4D(i, j, k, t);
    float dudx = ( ( UA4D(i, j, k, t) - UA4D(i-1, j, k, t) )/grid->dx) * UF(i);
    float dvdy = ( ( VA4D(i, j, k, t) - VA4D(i, j-1, k, t) )/grid->dy) * VF(j);

    buf0 = zvort_stretch;
    BUF4D(i, j, k, t) = -1.0*zv*( dudx + dvdy);
}

/* Compute the X vorticity tendency due to the turbulence closure scheme */
__device__ void calc_xvortturb_ten(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->turbv_4d_chunk;
    float *wstag = data->turbw_4d_chunk;
    float *dum0 = data->tem1_4d_chunk;

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/grid->dy ) * VF(j);
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/grid->dz ) * MF(k);
    TEM4D(i, j, k, t) = dwdy - dvdz; 
}

/* Compute the Y vorticity tendency due to the turbulence closure scheme */
__device__ void calc_yvortturb_ten(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->turbu_4d_chunk;
    float *wstag = data->turbw_4d_chunk;
    float *dum0 = data->tem2_4d_chunk;

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/grid->dx ) * UF(i);
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/grid->dz ) * MF(k);
    TEM4D(i, j, k, t) = dudz - dwdx;
}

/* Compute the Z vorticity tendency due to the turbulence closure scheme */
__device__ void calc_zvortturb_ten(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->turbu_4d_chunk;
    float *vstag = data->turbv_4d_chunk;
    float *dum0 = data->tem3_4d_chunk;

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/grid->dx) * UF(i);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/grid->dy) * VF(j);
    TEM4D(i, j, k, t) = dvdx - dudy;
}

__device__ void calc_zvort_solenoid(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // We can use p' here with no problems
    float *dum0 = data->pres_4d_chunk;
    // dP/dx
    float dpdx = ( (TEM4D(i+1, j, k, t) - TEM4D(i-1, j, k, t)) / ( 2*grid->dx ) ) * UH(i);
    // dP/dy
    float dpdy = ( (TEM4D(i, j+1, k, t) - TEM4D(i, j-1, k, t)) / ( 2*grid->dy ) ) * VH(j);

    dum0 = data->rho_4d_chunk;
    // dRho/dy
    // We use k-1 for the base state grid because it does not
    // have a lower ghost zone, so 0 corresponds to the surface 
    // instead of the ghost zone value
    float rho2 = TEM4D(i, j+1, k, t) + grid->rho0[k-1];
    float rho1 = TEM4D(i, j-1, k, t) + grid->rho0[k-1];
    float dalphady = ( ( (1./rho2) - (1./rho1) ) / ( 2*grid->dy ) ) * VH(j);

    // dRho/dx
    rho2 = TEM4D(i+1, j, k, t) + grid->rho0[k-1];
    rho1 = TEM4D(i-1, j, k, t) + grid->rho0[k-1];
    float dalphadx = ( ( (1./rho2) - (1./rho1) ) / ( 2*grid->dx ) ) * UH(i);

    // compute and save to the array
    float *buf0 = data->zvort_solenoid_4d_chunk; 
    BUF4D(i, j, k, t) = (dpdx*dalphady) - (dpdy*dalphadx);
}

/* When doing the parcel trajectory integration, George Bryan does
   some fun stuff with the lower boundaries/ghost zones of the arrays, presumably
   to prevent the parcels from exiting out the bottom of the domain
   or experience artificial values. This sets the ghost zone values. */
__global__ void applyMomentumBC(float *ustag, float *vstag, float *wstag, int NX, int NY, int NZ, int tStart, int tEnd) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    // this is done for easy comparison to CM1 code
    int ni = NX; int nj = NY;

    // this is a lower boundary condition, so only when k is 0
    // also this is on the u staggered mesh
    if (( j < nj+1) && ( i < ni+1) && ( k == 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // use the u stagger macro to handle the
            // proper indexing
            UA4D(i, j, 0, tidx) = UA4D(i, j, 1, tidx);
        }
    }
    
    // do the same but now on the v staggered grid
    if (( j < nj+1) && ( i < ni+1) && ( k == 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // use the v stagger macro to handle the
            // proper indexing
            VA4D(i, j, 0, tidx) = VA4D(i, j, 1, tidx);
        }
    }

    // do the same but now on the w staggered grid
    if (( j < nj+1) && ( i < ni+1) && ( k == 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // use the w stagger macro to handle the
            // proper indexing
            WA4D(i, j, 0, tidx) = -1*WA4D(i, j, 2, tidx);
        }
    }
}


__global__ void doTurbVort(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    // KELTON. FOR THE LOVE OF ALL THAT IS GOOD.
    // STOP CHANGING THE INDEX CHECK CONDITIONS. 
    // YOU'VE DONE THIS LIKE 5 TIMES NOW AND
    // CAUSE SEG FAULTS EVERY TIME. LEARN YOUR 
    // LESSON ALREADY. THIS WORKS. DON'T BREAK.
    // BAD.

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY+1) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvortturb_ten(grid, data, idx_4D, NX, NY, NZ);
        }
    }

    if ((i < NX+1) && (j < NY) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_yvortturb_ten(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i <= NX+1) && (j <= NY+1) && (k > 0) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvortturb_ten(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

__global__ void calcvort(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    // KELTON. FOR THE LOVE OF ALL THAT IS GOOD.
    // STOP CHANGING THE INDEX CHECK CONDITIONS. 
    // YOU'VE DONE THIS LIKE 5 TIMES NOW AND
    // CAUSE SEG FAULTS EVERY TIME. LEARN YOUR 
    // LESSON ALREADY. THIS WORKS. DON'T BREAK.
    // BAD.

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY+1) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort(grid, data, idx_4D, NX, NY, NZ);
        }
    }

    if ((i < NX+1) && (j < NY) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_yvort(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i <= NX+1) && (j <= NY+1) && (k > 0) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvort(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calcvortstretch(datagrid *grid, integration_data *data, int tStart, int tEnd) {
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
    if ((i < NX) && (j < NY+1) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort_stretch(grid, data, idx_4D, NX, NY, NZ);
        }
    }

    if ((i < NX+1) && (j < NY) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_yvort_stretch(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i < NX+1) && (j < NY+1) && (k > 0) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvort_stretch(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calcxvorttilt(datagrid *grid, integration_data *data, int tStart, int tEnd) {
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
    if ((i < NX+1) && (j < NY+1) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort_tilt(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calcyvorttilt(datagrid *grid, integration_data *data, int tStart, int tEnd) {
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
    if ((i < NX+1) && (j < NY+1) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_yvort_tilt(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the Vorticity Equation */
__global__ void calczvorttilt(datagrid *grid, integration_data *data, int tStart, int tEnd) {
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
    if ((i < NX+1) && (j < NY+1) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvort_tilt(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the pressure-volume solenoid term */
__global__ void calczvortsolenoid(datagrid *grid, integration_data *data, int tStart, int tEnd) {
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
    // Even though there are NZ points, it's a center difference
    // and we reach out NZ+1 points to get the derivatives
    if ((i < NX) && (j < NY) && (k < NZ) && ( i > 0 ) && (j > 0) && (k >= 1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvort_solenoid(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Zero out the temporary arrays */
__global__ void zeroTemArrays(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *dum0;
    if (( i < NX+1) && ( j < NY+1) && ( k < NZ+1)) {
        dum0 = data->tem1_4d_chunk;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem2_4d_chunk;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem3_4d_chunk;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem4_4d_chunk;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem5_4d_chunk;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
        dum0 = data->tem6_4d_chunk;
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            TEM4D(i, j, k, tidx) = 0.0;
        }
    }
}


/* Apply the free-slip lower boundary condition to the vorticity field. */
__global__ void applyVortBC(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *dum0;

    // NOTE: Not sure if need to use BUF4D or TEM4D. The size of the array
    // will for sure be respected by BUF4D but unsure if it even matters here.

    // This is a lower boundary condition, so only when k is 0.
    // Start with xvort. 
    if (( i < NX+1) && ( j < NY+1) && ( k == 1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // at this stage, xvort is in the tem1 array
            dum0 = data->tem1_4d_chunk;
            TEM4D(i, j, 1, tidx) = TEM4D(i, j, 2, tidx);
            // at this stage, yvort is in the tem2 array
            dum0 = data->tem2_4d_chunk;
            TEM4D(i, j, 1, tidx) = TEM4D(i, j, 2, tidx);
            // I'm technically ignoring an upper boundary condition
            // here, but we never really guarantee that we're at
            // the top of the model domain because we do a lot of subsetting.
            // So, for now, we assume we're nowehere near the top. 
        }
    }
}

/* Apply the free-slip lower boundary condition to the vorticity field. */
__global__ void applyVortTendBC(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *dum0;

    // NOTE: Not sure if need to use BUF4D or TEM4D. The size of the array
    // will for sure be respected by BUF4D but unsure if it even matters here.

    // This is a lower boundary condition, so only when k is 0.
    // Start with xvort. 
    if (( i < NX+1) && ( j < NY+1) && ( k == 1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1_4d_chunk;
            TEM4D(i, j, 1, tidx) = TEM4D(i, j, 2, tidx);
            dum0 = data->tem2_4d_chunk;
            TEM4D(i, j, 1, tidx) = TEM4D(i, j, 2, tidx);
            dum0 = data->tem3_4d_chunk;
            TEM4D(i, j, 1, tidx) = TEM4D(i, j, 2, tidx);
            dum0 = data->tem4_4d_chunk;
            TEM4D(i, j, 1, tidx) = TEM4D(i, j, 2, tidx);
            // I'm technically ignoring an upper boundary condition
            // here, but we never really guarantee that we're at
            // the top of the model domain because we do a lot of subsetting.
            // So, for now, we assume we're nowehere near the top. 
        }
    }
}

/* Average our vorticity values back to the scalar grid for interpolation
   to the parcel paths. We're able to do this in parallel by making use of
   the three temporary arrays allocated on our grid, which means that the
   xvort/yvort/zvort arrays will be averaged into tem1/tem2/tem3. After
   calling this kernel, you MUST set the new pointers appropriately. */
__global__ void doVortAvg(datagrid *grid, integration_data *data, int tStart, int tEnd) {

    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;

    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // average the temporary arrays into the result arrays
            dum0 = data->tem1_4d_chunk;
            buf0 = data->xvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem2_4d_chunk;
            buf0 = data->yvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem3_4d_chunk;
            buf0 = data->zvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );
        }
    }
}

__global__ void doTurbVortAvg(datagrid *grid, integration_data *data, int tStart, int tEnd) {

    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;

    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // average the temporary arrays into the result arrays
            dum0 = data->tem1_4d_chunk;
            buf0 = data->turbxvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem2_4d_chunk;
            buf0 = data->turbyvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem3_4d_chunk;
            buf0 = data->turbzvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );
        }
    }
}

/* Average the derivatives within the temporary arrays used to compute
   the tilting rate and then combine the terms into the final xvtilt
   array. It is assumed that the derivatives have been precomputed into
   the temporary arrays. */
__global__ void doXVortTiltAvg(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;
    float dudy, dwdx, dudz, dvdx;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1_4d_chunk;
            dudy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            dum0 = data->tem2_4d_chunk;
            dwdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem3_4d_chunk;
            dudz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem4_4d_chunk;
            dvdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            buf0 = data->xvtilt_4d_chunk;
            BUF4D(i, j, k, tidx) = -1.0*((dudy*dwdx) - (dudz*dvdx));
        }
    }
}

/* Average the derivatives within the temporary arrays used to compute
   the tilting rate and then combine the terms into the final yvtilt
   array. It is assumed that the derivatives have been precomputed into
   the temporary arrays. */
__global__ void doYVortTiltAvg(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;
    float dvdx, dwdy, dvdz, dudy;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1_4d_chunk;
            dvdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            dum0 = data->tem2_4d_chunk;
            dwdy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem3_4d_chunk;
            dvdz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem4_4d_chunk;
            dudy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            buf0 = data->yvtilt_4d_chunk;
            BUF4D(i, j, k, tidx) = -1.0*((dvdx*dwdy) - (dvdz*dudy));
        }
    }
}

/* Average the derivatives within the temporary arrays used to compute
   the tilting rate and then combine the terms into the final zvtilt
   array. It is assumed that the derivatives have been precomputed into
   the temporary arrays. */
__global__ void doZVortTiltAvg(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our grid indices based on our block and thread info
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float *buf0, *dum0;
    float dwdx, dvdz, dwdy, dudz;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1_4d_chunk;
            dwdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem2_4d_chunk;
            dvdz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem3_4d_chunk;
            dwdy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem4_4d_chunk;
            dudz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            buf0 = data->zvtilt_4d_chunk;
            BUF4D(i, j, k, tidx) = (dwdy*dudz)-(dwdx*dvdz);
        }
    }
}


#endif
