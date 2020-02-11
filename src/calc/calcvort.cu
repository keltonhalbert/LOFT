#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#ifndef VORT_CALC
#define VORT_CALC

/* Compute the nondimensional pressure */
__device__ void calc_pipert(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    // this is actually the pressure
    // perturbation, not the full pressure
    float *buf0 = data->prespert;
    float p = BUF4D(i, j, k, t)*100 + grid->p0[k]; 
    buf0 = data->pipert;
    BUF4D(i, j, k, t) = pow( p / 100000., 0.28571426) - pow( grid->p0[k] / 100000., 0.28571426);
}

/* Compute the x component of vorticity. After this is called by the calvort kernel, you must also run 
   the kernel for applying the lower boundary condition and then the kernel for averaging to the
   scalar grid. */
__device__ void calc_xvort(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->vstag;
    float *wstag = data->wstag;
    float *dum0 = data->tem1;
    float dy = yf(j) - yf(j-1);
    float dz = zf(k) - zf(k-1);

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/dy );
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dwdy - dvdz; 
    if (k == 1) {
        TEM4D(i, j, 0, t) = dwdy - dvdz; 
    }
}

/* Compute the y component of vorticity. After this is called by the calvort kernel, you must also run 
   the kernel for applying the lower boundary condition and then the kernel for averaging to the
   scalar grid. */
__device__ void calc_yvort(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->ustag;
    float *wstag = data->wstag;
    float *dum0 = data->tem2;
    float dx = xf(i) - xf(i-1);
    float dz = zf(k) - zf(k-1);

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/dx );
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dudz - dwdx;
    if (k == 1) {
        TEM4D(i, j, 0, t) = dudz - dwdx;
    }
}

/* Compute the z component of vorticity. After this is called by the calvort kernel, you must also run 
   the kernel for applying the lower boundary condition and then the kernel for averaging to the
   scalar grid. */
__device__ void calc_zvort(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->ustag;
    float *vstag = data->vstag;
    float *dum0 = data->tem3;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/dx);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/dy);
    TEM4D(i, j, k, t) = dvdx - dudy;
}

/* Compute the X component of vorticity tendency due
   to tilting Y and Z components into the X direction */
__device__ void calc_xvort_tilt(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->ustag;
    float *dum0;
    float dy = yf(j) - yf(j-1);
    float dz = zf(k) - zf(k-1);

    if (k >= 0) {
        // dudy in tem1
        dum0 = data->tem1;
        TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) ) / dy );
    }

    if (k >= 1) {
        // dudz in tem2
        dum0 = data->tem2;
        TEM4D(i, j, k, t) = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) ) / dz );
    }
    // This is the equivalent of our zero strain lower boundary
    if (k == 1) {
        // dudz in tem2
        dum0 = data->tem2;
        TEM4D(i, j, 0, t) = ( ( UA4D(i, j, 1, t) - UA4D(i, j, 0, t) ) / dz );
    }    
}

/* Compute the Y component of vorticity tendency due
   to tilting X and Z components into the X direction */
__device__ void calc_yvort_tilt(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->vstag;
    float *dum0;
    float dx = xf(i) - xf(i-1);
    float dz = zf(k) - zf(k-1);
    
    if (k >=0) {
        // dvdx in tem1
        dum0 = data->tem1;
        TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) ) / dx );
    }

    if (k >= 1) {
        // dvdz in tem2
        dum0 = data->tem2;
        TEM4D(i, j, k, t) = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) ) / dz );
    }
    // This is the equivalent of our zero strain lower boundary
    if (k == 1) {
        // dvdz in tem2
        dum0 = data->tem2;
        TEM4D(i, j, 0, t) = ( ( VA4D(i, j, 1, t) - VA4D(i, j, 0, t) ) / dz );
    }
}

/* Compute the Z component of vorticity tendency due
   to tilting X and Y components into the X direction */
__device__ void calc_zvort_tilt(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *wstag = data->wstag;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);

    // Compute dw/dx and put it in the tem1 array. The derivatives
    // land on weird places so we have to average each derivative back
    // to the scalar grid, resulting in this clunky approach
    if (k >= 0) {
        float *dum0 = data->tem1;
        TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) ) / dx );

        // put dw/dy in tem2
        dum0 = data->tem2;
        TEM4D(i, j, k, t) = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) ) / dy );
    }
}

/* Compute the X component of vorticity tendency due
   to stretching of the vorticity along the X axis. */
__device__ void calc_xvort_stretch(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    

    float *wstag = data->wstag;
    float *vstag = data->vstag;
    float *xvort = data->xvort;
    float *xvort_stretch = data->xvstretch;
    float dy = yf(j+1) - yf(j);
    float dz = zf(k+1) - zf(k);

    float rrv = grid->rho0[k];
    float rrw1, rrw2;
    if ( k == 0 ) {
        rrw1 = 1.75*grid->rho0[1] - grid->rho0[2] + 0.25*grid->rho0[3];
        rrw2 = 0.5*grid->rho0[2] + 0.5*grid->rho0[3];
    }
    else {
        rrw1 = 0.5*grid->rho0[k-1] + 0.5*grid->rho0[k  ];
        rrw2 = 0.5*grid->rho0[k  ] + 0.5*grid->rho0[k+1];
    }

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = xvort;
    float xv = BUF4D(i, j, k, t);
    float dvdy, dwdz;
    dvdy = rrv * ( ( VA4D(i, j+1, k, t) - VA4D(i, j, k, t) )/dy);
    dwdz = ( (  rrw2*WA4D(i, j, k+1, t) - rrw1*WA4D(i, j, k, t) )/dz);

    buf0 = xvort_stretch;
    BUF4D(i, j, k, t) = -xv*( (dvdy + dwdz) / rrv);

}

/* Compute the Y component of vorticity tendency due
   to stretching of the vorticity along the Y axis. */
__device__ void calc_yvort_stretch(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->ustag;
    float *wstag = data->wstag;
    float *yvort = data->yvort;
    float *yvort_stretch = data->yvstretch;
    float dx = xf(i+1) - xf(i);
    float dz = zf(k+1) - zf(k);

    float rru = grid->rho0[k];
    float rrw1, rrw2;
    if ( k == 0 ) {
        rrw1 = 1.75*grid->rho0[1] - grid->rho0[2] + 0.25*grid->rho0[3];
        rrw2 = 0.5*grid->rho0[2] + 0.5*grid->rho0[3];
    }
    else {
        rrw1 = 0.5*grid->rho0[k-1] + 0.5*grid->rho0[k  ];
        rrw2 = 0.5*grid->rho0[k  ] + 0.5*grid->rho0[k+1];
    }

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = yvort;
    float yv = BUF4D(i, j, k, t);
    float dudx, dwdz;
    dudx = ( rru*( UA4D(i+1, j, k, t) - UA4D(i, j, k, t) )/dx);
    dwdz = ( ( rrw2*WA4D(i, j, k+1, t) - rrw1*WA4D(i, j, k, t) )/dz);

    buf0 = yvort_stretch;
    BUF4D(i, j, k, t) = -yv*( (dudx + dwdz) / rru);
}

/* Compute the Z component of vorticity tendency due
   to stretching of the vorticity along the Z axis. */
__device__ void calc_zvort_stretch(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->ustag;
    float *vstag = data->vstag;
    float *wstag = data->wstag;
    float *zvort = data->zvort;
    float *zvort_stretch = data->zvstretch;
    float dx = xf(i+1) - xf(i);
    float dy = yf(j+1) - yf(j);

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
__device__ void calc_xvortturb_ten(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->turbv;
    float *wstag = data->turbw;
    float *dum0 = data->tem1;
    float dy = yf(j) - yf(j-1);
    float dz = zf(k) - zf(k-1);

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/dy );
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dwdy - dvdz; 
    if (k == 1) {
        TEM4D(i, j, 0, t) = dwdy - dvdz; 
    }
}

/* Compute the Y vorticity tendency due to the turbulence closure scheme */
__device__ void calc_yvortturb_ten(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->turbu;
    float *wstag = data->turbw;
    float *dum0 = data->tem2;
    float dx = xf(i) - xf(i-1);
    float dz = zf(k) - zf(k-1);

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/dx );
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dudz - dwdx;
    if (k == 1) {
        TEM4D(i, j, 0, t) = dudz - dwdx;
    }
}

/* Compute the Z vorticity tendency due to the turbulence closure scheme */
__device__ void calc_zvortturb_ten(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->turbu;
    float *vstag = data->turbv;
    float *dum0 = data->tem3;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/dx);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/dy);
    TEM4D(i, j, k, t) = dvdx - dudy;
}


/* Compute the X vorticity tendency due to the 6th order numerical diffusion */
__device__ void calc_xvortdiff_ten(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *vstag = data->diffv;
    float *wstag = data->diffw;
    float *dum0 = data->tem1;
    float dy = yf(j) - yf(j-1);
    float dz = zf(k) - zf(k-1);

    float dwdy = ( ( WA4D(i, j, k, t) - WA4D(i, j-1, k, t) )/dy );
    float dvdz = ( ( VA4D(i, j, k, t) - VA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dwdy - dvdz; 
    if (k == 1) {
        TEM4D(i, j, 0, t) = dwdy - dvdz; 
    }
}

/* Compute the Y vorticity tendency due to the 6th order numerical diffusion */
__device__ void calc_yvortdiff_ten(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->diffu;
    float *wstag = data->diffw;
    float *dum0 = data->tem2;
    float dx = xf(i) - xf(i-1);
    float dz = zf(k) - zf(k-1);

    float dwdx = ( ( WA4D(i, j, k, t) - WA4D(i-1, j, k, t) )/dx );
    float dudz = ( ( UA4D(i, j, k, t) - UA4D(i, j, k-1, t) )/dz );
    TEM4D(i, j, k, t) = dudz - dwdx;
    if (k == 1) {
        TEM4D(i, j, 0, t) = dudz - dwdx;
    }
}

/* Compute the Z vorticity tendency due to the 6th order numerical diffusion */
__device__ void calc_zvortdiff_ten(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *ustag = data->diffu;
    float *vstag = data->diffv;
    float *dum0 = data->tem3;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);

    float dvdx = ( ( VA4D(i, j, k, t) - VA4D(i-1, j, k, t) )/dx);
    float dudy = ( ( UA4D(i, j, k, t) - UA4D(i, j-1, k, t) )/dy);
    TEM4D(i, j, k, t) = dvdx - dudy;
}

__device__ void calc_xvort_baro(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    const float reps = 461.5 / 287.04;
    const float g = 9.81;

    float dy = yf(j) - yf(j-1);

    float *buf0 = data->thrhopert;
    float qvbar1 = grid->qv0[k];
    float thbar1 = grid->th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    // dthrho/dy
    float dthdy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    // compute and save to the array
    buf0 = data->xvort_baro; 
    BUF4D(i, j, k, t) = (g/thbar1)*dthdy; 
    if (k == 1) {
        // the d/dy terms are defined at k = 1,
        // go get those
        dthdy = ( (BUF4D(i, j+1, 0, t) - BUF4D(i, j-1, 0, t)) / ( 2*dy ) );
        BUF4D(i, j, 0, t) = (g/thbar1)*dthdy; 
    }
}

__device__ void calc_yvort_baro(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    const float reps = 461.5 / 287.04;
    const float g = 9.81;

    float dx = xf(j) - xf(j-1);

    float *buf0 = data->thrhopert;
    float qvbar1 = grid->qv0[k];
    float thbar1 = grid->th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    // dthrho/dy
    float dthdx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );

    // compute and save to the array
    buf0 = data->yvort_baro; 
    BUF4D(i, j, k, t) = (g/thbar1)*dthdx; 
    if (k == 1) {
        // the d/dy terms are defined at k = 1,
        // go get those
        dthdx = ( (BUF4D(i+1, j, 0, t) - BUF4D(i-1, j, 0, t)) / ( 2*dx ) );
        BUF4D(i, j, 0, t) = -1.*(g/thbar1)*dthdx; 
    }
}
__device__ void calc_xvort_solenoid(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float cp = 1005.7;
    const float reps = 461.5 / 287.04;

    float dy = yf(j) - yf(j-1);
    float dz = zf(k) - zf(k-1);

    float *buf0 = data->pipert;
    // dPi/dz
    float pi_upper = BUF4D(i, j, k+1, t);
    float pi_lower = BUF4D(i, j, k-1, t);
    float dpidz = ( (pi_upper - pi_lower) / ( 2*dz ) );
    // dPi/dy
    float dpidy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    buf0 = data->thrhopert;
    float qvbar1 = grid->qv0[k+1];
    float qvbar2 = grid->qv0[k-1];
    float thbar1 = grid->th0[k+1]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thbar2 = grid->th0[k-1]*(1.0+reps*qvbar2)/(1.0+qvbar2); 
    // dthrho/dy
    float dthdy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    // dthrho/dz
    float dthdz = ( ((BUF4D(i, j, k+1, t) + thbar1) - (BUF4D(i, j, k-1, t) + thbar2)) / ( 2*dz ) );

    // compute and save to the array
    buf0 = data->xvort_solenoid; 
    BUF4D(i, j, k, t) = -cp*(dthdy*dpidz - dthdz*dpidy); 
    if (k == 1) {
        // the d/dy terms are defined at k = 1,
        // go get those
        dpidy = ( ( BUF4D(i, j+1, 0, t) - BUF4D(i, j-1, 0, t) ) / (2*dy) );
        dthdy = ( (BUF4D(i, j+1, 0, t) - BUF4D(i, j-1, 0, t)) / ( 2*dy ) );
        BUF4D(i, j, 0, t) = -cp*(dthdy*dpidz - dthdz*dpidy); 
    }
}

__device__ void calc_yvort_solenoid(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float cp = 1005.7;
    const float reps = 461.5 / 287.04;
    float dx = xf(i) - xf(i-1);
    float dz = zf(k) - zf(k-1);

    float *buf0 = data->pipert;
    // dPi/dz
    float pi_upper = BUF4D(i, j, k+1, t);
    float pi_lower = BUF4D(i, j, k-1, t);
    float dpidz = ( (pi_upper - pi_lower) / ( 2*dz ) );
    // dPi/dx
    float dpidx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );

    buf0 = data->thrhopert;
    float qvbar1 = grid->qv0[k+1];
    float qvbar2 = grid->qv0[k-1];
    float thbar1 = grid->th0[k+1]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thbar2 = grid->th0[k-1]*(1.0+reps*qvbar2)/(1.0+qvbar2); 
    // dthrho/dx
    float dthdx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );

    // dthrho/dz
    float dthdz = ( ((BUF4D(i, j, k+1, t) + thbar1) - (BUF4D(i, j, k-1, t) + thbar2)) / ( 2*dz ) );

    // compute and save to the array
    buf0 = data->yvort_solenoid; 
    BUF4D(i, j, k, t) = -cp*(dthdz*dpidx - dthdx*dpidz); 
    if (k == 1) {
        // the d/dx terms are defined at k = 1,
        // go get those
        dpidx = ( (BUF4D(i+1, j, 0, t) - BUF4D(i-1, j, 0, t)) / ( 2*dx ) );
        dthdx = ( (BUF4D(i+1, j, 0, t) - BUF4D(i-1, j, 0, t)) / ( 2*dx ) );
        BUF4D(i, j, 0, t) = -cp*(dthdz*dpidx - dthdx*dpidz); 
    }
}

__device__ void calc_zvort_solenoid(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float cp = 1005.7;
    float dx = xf(i) - xf(i-1);
    float dy = yf(j) - yf(j-1);

    float *buf0 = data->pipert;
    // dPi/dx
    float dpidx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );
    // dPi/dy
    float dpidy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    buf0 = data->thrhopert;
    // dthrho/dx
    float dthdx = ( (BUF4D(i+1, j, k, t) - BUF4D(i-1, j, k, t)) / ( 2*dx ) );

    // dthrho/dy
    float dthdy = ( (BUF4D(i, j+1, k, t) - BUF4D(i, j-1, k, t)) / ( 2*dy ) );

    // compute and save to the array
    buf0 = data->yvort_solenoid; 
    BUF4D(i, j, k, t) = -cp*(dthdx*dpidy - dthdy*dpidx); 
}

#endif
