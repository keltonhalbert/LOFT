#include <iostream>
#include <stdio.h>
#include "datastructs.cu"
#include "macros.cpp"
#include "interp.cu"
#ifndef VORT_CU
#define VORT_CU

/* Compute the nondimensional pressure */
__device__ void calc_pi(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // this is actually the pressure
    // perturbation, not the full pressure
    float *buf0 = data->pres_4d_chunk;
    float p = BUF4D(i, j, k, t) + grid->p0[k-1]; 
    buf0 = data->pi_4d_chunk;
    BUF4D(i, j, k, t) = powf( p / 100000., 0.28571426);
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
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/dy );
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dwdy - dvdz; 
    if (k == 2) {
        TEM4D(i, j, 1, t) = dwdy - dvdz; 
    }
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
    float dx = grid->xf[i+1] - grid->xf[i];
    float dz = grid->zf[k+1] - grid->zf[k];

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/dx );
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dudz - dwdx;
    if (k == 2) {
        TEM4D(i, j, 1, t) = dudz - dwdx;
    }
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
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/dx);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/dy);
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
    float *dum0;
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    if (k >= 1) {
        // dudy in tem1
        dum0 = data->tem1_4d_chunk;
        TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) ) / dy );
    }

    if (k >= 2) {
        // dudz in tem2
        dum0 = data->tem2_4d_chunk;
        TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) ) / dz );
    }
    // This is the equivalent of our zero strain lower boundary
    if (k == 2) {
        // dudz in tem2
        dum0 = data->tem2_4d_chunk;
        TEM4D(i, j, 1, t) = ( ( UA4D(i, j, 2, t) - UA4D(i, j, 1, t) ) / dz );
    }    
}

/* Compute the Y component of vorticity tendency due
   to tilting X and Z components into the X direction */
__device__ void calc_yvort_tilt(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->v_4d_chunk;
    float *dum0;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dz = grid->zf[k+1] - grid->zf[k];
    
    if (k >=1) {
        // dvdx in tem1
        dum0 = data->tem1_4d_chunk;
        TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) ) / dx );
    }

    if (k >= 2) {
        // dvdz in tem2
        dum0 = data->tem2_4d_chunk;
        TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) ) / dz );
    }
    // This is the equivalent of our zero strain lower boundary
    if (k == 2) {
        // dvdz in tem2
        dum0 = data->tem2_4d_chunk;
        TEM4D(i, j, 1, t) = ( ( VA4D(i, j, 2, t) - VA4D(i, j, 1, t) ) / dz );
    }
}

/* Compute the Z component of vorticity tendency due
   to tilting X and Y components into the X direction */
__device__ void calc_zvort_tilt(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *wstag = data->w_4d_chunk;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];

    // Compute dw/dx and put it in the tem1 array. The derivatives
    // land on weird places so we have to average each derivative back
    // to the scalar grid, resulting in this clunky approach
    if (k >= 1) {
        float *dum0 = data->tem1_4d_chunk;
        TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) ) / dx );

        // put dw/dy in tem2
        dum0 = data->tem2_4d_chunk;
        TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) ) / dy );
    }
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
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = xvort;
    float xv = BUF4D(i, j, k, t);
    float dvdy, dwdz;
    dvdy = ( ( VA4D(i, j+1, k, t) - VA4D(i, j, k, t) )/dy);
    dwdz = ( ( WA4D(i, j, k+1, t) - WA4D(i, j, k, t) )/dz);

    buf0 = xvort_stretch;
    BUF4D(i, j, k, t) = -xv*( dvdy + dwdz);

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
    float dx = grid->xf[i+1] - grid->xf[i];
    float dz = grid->zf[k+1] - grid->zf[k];

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = yvort;
    float yv = BUF4D(i, j, k, t);
    float dudx, dwdz;
    dudx = ( ( UA4D(i+1, j, k, t) - UA4D(i, j, k, t) )/dx);
    dwdz = ( ( WA4D(i, j, k+1, t) - WA4D(i, j, k, t) )/dz);

    buf0 = yvort_stretch;
    BUF4D(i, j, k, t) = -yv*( dudx + dwdz);
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
    float *wstag = data->w_4d_chunk;
    float *zvort = data->zvort_4d_chunk;
    float *zvort_stretch = data->zvstretch_4d_chunk;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = zvort;
    float zv = BUF4D(i, j, k, t);
    float dudx = ( ( UA4D(i+1, j, k, t) - UA4D(i, j, k, t) )/dx);
    float dvdy = ( ( VA4D(i, j+1, k, t) - VA4D(i, j, k, t) )/dy);

    buf0 = zvort_stretch;
    BUF4D(i, j, k, t) = -zv*( dudx + dvdy);
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
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/dy );
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dwdy - dvdz; 
    if (k == 2) {
        TEM4D(i, j, 1, t) = dwdy - dvdz; 
    }
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
    float dx = grid->xf[i+1] - grid->xf[i];
    float dz = grid->zf[k+1] - grid->zf[k];

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/dx );
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dudz - dwdx;
    if (k == 2) {
        TEM4D(i, j, 1, t) = dudz - dwdx;
    }
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
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/dx);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/dy);
    TEM4D(i, j, k, t) = dvdx - dudy;
}


/* Compute the X vorticity tendency due to the 6th order numerical diffusion */
__device__ void calc_xvortdiff_ten(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->diffv_4d_chunk;
    float *wstag = data->diffw_4d_chunk;
    float *dum0 = data->tem1_4d_chunk;
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/dy );
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dwdy - dvdz; 
    if (k == 2) {
        TEM4D(i, j, 1, t) = dwdy - dvdz; 
    }
}

/* Compute the Y vorticity tendency due to the 6th order numerical diffusion */
__device__ void calc_yvortdiff_ten(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->diffu_4d_chunk;
    float *wstag = data->diffw_4d_chunk;
    float *dum0 = data->tem2_4d_chunk;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dz = grid->zf[k+1] - grid->zf[k];

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/dx );
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dudz - dwdx;
    if (k == 2) {
        TEM4D(i, j, 1, t) = dudz - dwdx;
    }
}

/* Compute the Z vorticity tendency due to the 6th order numerical diffusion */
__device__ void calc_zvortdiff_ten(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->diffu_4d_chunk;
    float *vstag = data->diffv_4d_chunk;
    float *dum0 = data->tem3_4d_chunk;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/dx);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/dy);
    TEM4D(i, j, k, t) = dvdx - dudy;
}

__device__ void calc_xvort_solenoid(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float cp = 1005.7;
    float dy = grid->yf[j+1] - grid->yf[j];
    float dz = grid->zf[k+1] - grid->zf[k];

    float *buf0 = data->pi_4d_chunk;
    // dPi/dz
    float pi_upper = BUF4D(i, j, k+1, t);
    float pi_lower = BUF4D(i, j, k-1, t);
    float dpidz = ( (pi_upper - pi_lower) / ( 2*dz ) );
    // dPi/dy
    float dpidy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    buf0 = data->th_4d_chunk;
    // dthrho/dy
    float dthdy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    // ddthrho/dz
    float dthdz = ( (BUF4D(i, j, k+1, t) - BUF4D(i, j, k-1, t)) / ( 2*dz ) );

    // compute and save to the array
    buf0 = data->xvort_solenoid_4d_chunk; 
    BUF4D(i, j, k, t) = -cp*(dthdy*dpidz - dthdz*dpidy); 
    if (k == 2) {
        // the d/dy terms are defined at k = 1,
        // go get those
        dpidy = ( ( BUF4D(i, j+1, 1, t) - BUF4D(i, j-1, 1, t) ) / (2*dy) );
        dthdy = ( (BUF4D(i, j+1, 1, t) - BUF4D(i, j-1, 1, t)) / ( 2*dy ) );
        BUF4D(i, j, 1, t) = -cp*(dthdy*dpidz - dthdz*dpidy); 
    }
}

__device__ void calc_yvort_solenoid(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float cp = 1005.7;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dz = grid->zf[k+1] - grid->zf[k];

    float *buf0 = data->pi_4d_chunk;
    // dPi/dz
    // We need pres0 because we're doing a vertical derivative
    float pi_upper = BUF4D(i, j, k+1, t);
    float pi_lower = BUF4D(i, j, k-1, t);
    float dpidz = ( (pi_upper - pi_lower) / ( 2*dz ) );
    // dPi/dx
    float dpidx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );

    buf0 = data->th_4d_chunk;
    // dthrho/dx
    float dthdx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );

    // dthrho/dz
    float dthdz = ( (BUF4D(i, j, k+1, t) - BUF4D(i, j, k-1, t)) / ( 2*dz ) );

    // compute and save to the array
    buf0 = data->yvort_solenoid_4d_chunk; 
    BUF4D(i, j, k, t) = -cp*(dthdz*dpidx - dthdx*dpidz); 
    if (k == 2) {
        // the d/dx terms are defined at k = 1,
        // go get those
        dpidx = ( (BUF4D(i+1, j, 1, t) - BUF4D(i-1, j, 1, t)) / ( 2*dx ) );
        dthdx = ( (BUF4D(i+1, j, 1, t) - BUF4D(i-1, j, 1, t)) / ( 2*dx ) );
        BUF4D(i, j, 1, t) = -cp*(dthdz*dpidx - dthdx*dpidz); 
    }
}

__device__ void calc_zvort_solenoid(datagrid *grid, integration_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float cp = 1005.7;
    float dx = grid->xf[i+1] - grid->xf[i];
    float dy = grid->yf[j+1] - grid->yf[j];

    float *buf0 = data->pi_4d_chunk;
    // dPi/dx
    float dpidx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );
    // dPi/dy
    float dpidy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    buf0 = data->th_4d_chunk;
    // dthrho/dx
    float dthdx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );

    // dthrho/dy
    float dthdy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    // compute and save to the array
    buf0 = data->yvort_solenoid_4d_chunk; 
    BUF4D(i, j, k, t) = -cp*(dthdx*dpidy - dthdy*dpidx); 
    /*
    if (k == 2) {
        BUF4D(i, j, 1, t) = -cp*(dthdx*dpidy - dthdy*dpidx); 
    }
    */
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

__global__ void doDiffVort(datagrid *grid, integration_data *data, int tStart, int tEnd) {
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
            calc_xvortdiff_ten(grid, data, idx_4D, NX, NY, NZ);
        }
    }

    if ((i < NX+1) && (j < NY) && (k > 1) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_yvortdiff_ten(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i <= NX+1) && (j <= NY+1) && (k > 0) && (k < NZ+1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvortdiff_ten(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

__global__ void calcpi(datagrid *grid, integration_data *data, int tStart, int tEnd) {
    // get our 3D index based on our blocks/threads
    int i = (blockIdx.x*blockDim.x) + threadIdx.x;
    int j = (blockIdx.y*blockDim.y) + threadIdx.y;
    int k = (blockIdx.z*blockDim.z) + threadIdx.z;
    int idx_4D[4];
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < NX) && (j < NY) && (k >= 1) && (k < NZ+1) && (i > 0) && (j > 0)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_pi(grid, data, idx_4D, NX, NY, NZ);
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
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort_stretch(grid, data, idx_4D, NX, NY, NZ);
        }
    }

    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_yvort_stretch(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
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
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
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
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
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
    if ((i < NX) && (j < NY) && (k > 0) && (k < NZ)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvort_tilt(grid, data, idx_4D, NX, NY, NZ);
        }
    }
}

/* Compute the forcing tendencies from the pressure-volume solenoid term */
__global__ void calcvortsolenoid(datagrid *grid, integration_data *data, int tStart, int tEnd) {
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
    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k >= 1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_zvort_solenoid(grid, data, idx_4D, NX, NY, NZ);
        }
    }
    if ((i < NX-1) && (j < NY-1) && (k < NZ) && ( i > 0 ) && (j > 0) && (k > 1)) {
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort_solenoid(grid, data, idx_4D, NX, NY, NZ);
            calc_yvort_solenoid(grid, data, idx_4D, NX, NY, NZ);
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


__global__ void doDiffVortAvg(datagrid *grid, integration_data *data, int tStart, int tEnd) {

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
            buf0 = data->diffxvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            dum0 = data->tem2_4d_chunk;
            buf0 = data->diffyvort_4d_chunk;
            BUF4D(i, j, k, tidx) = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) +\
                                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem3_4d_chunk;
            buf0 = data->diffzvort_4d_chunk;
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
    float dudy,dudz;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1_4d_chunk;
            //dudy = TEM4D(i, j, k, tidx);
            dudy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            dum0 = data->tem2_4d_chunk;
            //dudz = TEM4D(i, j, k, tidx);
            dudz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i+1, j, k+1, tidx) );

            buf0 = data->zvort_4d_chunk;
            float zvort = BUF4D(i, j, k, tidx);
            buf0 = data->yvort_4d_chunk;
            float yvort = BUF4D(i, j, k, tidx);

            buf0 = data->xvtilt_4d_chunk;
            BUF4D(i, j, k, tidx) = zvort * dudz + yvort * dudy; 
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
    float dvdx, dvdz;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1_4d_chunk;
            //dvdx = TEM4D(i, j, k, tidx);
            dvdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j+1, k, tidx) );

            dum0 = data->tem2_4d_chunk;
            //dvdz = TEM4D(i, j, k, tidx);
            dvdz = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );

            buf0 = data->xvort_4d_chunk;
            float xvort = BUF4D(i, j, k, tidx);
            buf0 = data->zvort_4d_chunk;
            float zvort = BUF4D(i, j, k, tidx);

            buf0 = data->yvtilt_4d_chunk;
            BUF4D(i, j, k, tidx) = xvort * dvdx + zvort * dvdz; 
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
    float dwdx, dwdy;

    // We do the average for each array at a given point
    // and then finish the computation for the zvort tilt
    if ((i < NX) && (j < NY) && (k < NZ) && (k > 0)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            dum0 = data->tem1_4d_chunk;
            //dwdx = TEM4D(i, j, k, tidx);
            dwdx = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i+1, j, k, tidx) + \
                            TEM4D(i, j+1, k, tidx) + TEM4D(i+1, j, k+1, tidx) );

            dum0 = data->tem2_4d_chunk;
            dwdy = 0.25 * ( TEM4D(i, j, k, tidx) + TEM4D(i, j+1, k, tidx) + \
                            TEM4D(i, j, k+1, tidx) + TEM4D(i, j+1, k+1, tidx) );
            //dwdy = TEM4D(i, j, k, tidx);
            buf0 = data->xvort_4d_chunk;
            float xvort = BUF4D(i, j, k, tidx);
            buf0 = data->yvort_4d_chunk;
            float yvort = BUF4D(i, j, k, tidx);
            
            buf0 = data->zvtilt_4d_chunk;
            BUF4D(i, j, k, tidx) = xvort * dwdx + yvort * dwdy; 
        }
    }
}
#endif
