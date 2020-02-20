#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/constants.h"
#include "../include/macros.h"
#ifndef VORT_CALC
#define VORT_CALC

/* Compute the nondimensional pressure perturbation given
   the pressure perturbation field and base state pressure. 
   Due to the nature of exponentials, both are required to first
   compute the total nondimensional pressure and then convert it
   to a perturbation. 
 
   NOTE: Units of p0 have been changed in 
   more recent datasets and this will need 
   to be modified appropriately. 

   INPUT
   prespert: hPa
   p0: Pa

   RETURNS
   pipert: unitless
 */
__device__ void calc_pipert(float *prespert, float *p0, float *pipert, int i, int j, int k, int NX, int NY) {
    float *buf0 = prespert; 
    float p = BUF(i, j, k)*100 + p0[k]; ; // convert from hPa to Pa 
    buf0 = pipert;
    BUF(i, j, k) = pow( p * rp00, rovcp) - pow( p0[k] * rp00, rovcp); 
}

/*  Compute the component of vorticity along the x-axis.
    This stencil does not handle averaging the values
    to the scalar grid, and dx and dy are passed as
    arguments to the stencil. 
   
    INPUT
    vstag: meters/second
    wstag: meters/second
    dx: meters
    dy: meters

    OUTPUT:
    xvort: 1/second
 */
__device__ void calc_xvort(float *vstag, float *wstag, float *xvort, float dy, float dz, int i, int j, int k, int NX, int NY) {
    float *dum0 = xvort;
    float dwdy = ( ( WA(i, j, k) - WA(i, j-1, k) )/dy );
    float dvdz = ( ( VA(i, j, k) - VA(i, j, k-1) )/dz );
    TEM(i, j, k) = dwdy - dvdz; 
}

/*  Compute the component of vorticity along the y-axis.
    This stencil does not handle averaging the values
    to the scalar grid, and dx and dy are passed as
    arguments to the stencil. 
   
    INPUT
    ustag: meters/second
    wstag: meters/second
    dx: meters
    dz: meters

    OUTPUT:
    yvort: 1/second
 */
__device__ void calc_yvort(float *ustag, float *wstag, float *yvort, float dx, float dz, int i, int j, int k, int NX, int NY) {
    float *dum0 = yvort;
    float dwdx = ( ( WA(i, j, k) - WA(i-1, j, k) )/dx );
    float dudz = ( ( UA(i, j, k) - UA(i, j, k-1) )/dz );
    TEM(i, j, k) = dudz - dwdx;
}

/*  Compute the component of vorticity along the z-axis.
    This stencil does not handle averaging the values
    to the scalar grid, and dx and dy are passed as
    arguments to the stencil. 
   
    INPUT
    ustag: meters/second
    vstag: meters/second
    dx: meters
    dy: meters

    OUTPUT:
    zvort: 1/second
 */
__device__ void calc_zvort(float *ustag, float *vstag, float *zvort, float dx, float dy, int i, int j, int k, int NX, int NY) {
    float *dum0 = zvort;
    float dvdx = ( ( VA(i, j, k) - VA(i-1, j, k) )/dx);
    float dudy = ( ( UA(i, j, k) - UA(i, j-1, k) )/dy);
    TEM(i, j, k) = dvdx - dudy;
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
__device__ void calc_xvort_stretch(float *vstag, float *wstag, float *xvort, float *xvort_stretch, \
                                   float dy, float dz, int i, int j, int k, int NX, int NY) {

    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = xvort;
    float xv = BUF(i, j, k);
    float dvdy, dwdz;
    dvdy = ( VA(i, j+1, k) - VA(i, j, k) )/dy;
    dwdz = ( WA(i, j, k+1) - WA(i, j, k) )/dz;

    buf0 = xvort_stretch;
    BUF(i, j, k) = -xv*( (dvdy + dwdz) );
}

/* Compute the Y component of vorticity tendency due
   to stretching of the vorticity along the Y axis. */
__device__ void calc_yvort_stretch(float *ustag, float *wstag, float *yvort, float *yvort_stretch, \
                                   float dx, float dz, int i, int j, int k, int NX, int NY) {
    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = yvort;
    float yv = BUF(i, j, k);
    float dudx, dwdz;
    dudx = ( UA(i+1, j, k) - UA(i, j, k) )/dx;
    dwdz = ( WA(i, j, k+1) - WA(i, j, k) )/dz;

    buf0 = yvort_stretch;
    BUF(i, j, k) = -yv*( (dudx + dwdz) );
}

/* Compute the Z component of vorticity tendency due
   to stretching of the vorticity along the Z axis. */
__device__ void calc_zvort_stretch(float *ustag, float *vstag, float *zvort, float *zvort_stretch, \
                                   float dx, float dy, int i, int j, int k, int NX, int NY) {
    // this stencil conveniently lands itself on the scalar grid,
    // so we won't have to worry about doing any averaging. I think.
    float *buf0 = zvort;
    float zv = BUF(i, j, k);
    float dudx = ( UA(i+1, j, k) - UA(i, j, k) )/dx;
    float dvdy = ( VA(i, j+1, k) - VA(i, j, k) )/dy;

    buf0 = zvort_stretch;
    BUF(i, j, k) = -zv*( dudx + dvdy);
}

__device__ void calc_xvort_baro(float *thrhopert, float *th0, float *qv0, float *xvort_baro, \
                                float dy, int i, int j, int k, int NX, int NY) {
    float *buf0 = thrhopert;
    float qvbar1 = qv0[k];
    float thbar1 = th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    // dthrho/dy
    float dthdy = ( (BUF(i, j+1, k) - BUF(i, j-1, k)) / ( dy ) );

    // compute and save to the array
    buf0 = xvort_baro; 
    BUF(i, j, k) = (g/thbar1)*dthdy; 
}

__device__ void calc_yvort_baro(float *thrhopert, float *th0, float *qv0, float *yvort_baro, \
                                float dx, int i, int j, int k, int NX, int NY) {
    float *buf0 = thrhopert;
    float qvbar1 = qv0[k];
    float thbar1 = th0[k]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    // dthrho/dy
    float dthdx = ( (BUF(i+1, j, k) - BUF(i-1, j, k)) / ( dx ) );

    // compute and save to the array
    buf0 = yvort_baro; 
    BUF(i, j, k) = (g/thbar1)*dthdx; 
}
__device__ void calc_xvort_solenoid(float *pipert, float *thrhopert, float *th0, float *qv0, float *xvort_solenoid, \
                                    float dy, float dz, int i, int j, int k, int NX, int NY) {
    float *buf0 = pipert;
    float dpidz = ( (BUF(i, j, k+1) - BUF(i, j, k-1)) / ( dz ) );
    float dpidy = ( (BUF(i, j+1, k) - BUF(i, j-1, k)) / ( dy ) );

    buf0 = thrhopert;
    float qvbar1 = qv0[k+1];
    float qvbar2 = qv0[k-1];
    float thbar1 = th0[k+1]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thbar2 = th0[k-1]*(1.0+reps*qvbar2)/(1.0+qvbar2); 
    float dthdy = ( (BUF(i, j+1, k) - BUF(i, j-1, k)) / ( dy ) );
    float dthdz = ( ((BUF(i, j, k+1) + thbar1) - (BUF(i, j, k-1) + thbar2)) / ( dz ) );

    // compute and save to the array
    buf0 = xvort_solenoid; 
    BUF(i, j, k) = -cp*(dthdy*dpidz - dthdz*dpidy); 
}

__device__ void calc_yvort_solenoid(float *pipert, float *thrhopert, float *th0, float *qv0, float *yvort_solenoid, \
                                    float dx, float dz, int i, int j, int k, int NX, int NY) {
    float *buf0 = pipert;
    float dpidz = ( (BUF(i, j, k+1) - BUF(i, j, k-1)) / ( dz ) );
    float dpidx = ( (BUF(i+1, j, k) - BUF(i-1, j, k)) / ( dx ) );

    buf0 = thrhopert;
    float qvbar1 = qv0[k+1];
    float qvbar2 = qv0[k-1];
    float thbar1 = th0[k+1]*(1.0+reps*qvbar1)/(1.0+qvbar1); 
    float thbar2 = th0[k-1]*(1.0+reps*qvbar2)/(1.0+qvbar2); 
    float dthdx = ( (BUF(i+1, j, k) - BUF(i-1, j, k)) / ( dx ) );
    float dthdz = ( ((BUF(i, j, k+1) + thbar1) - (BUF(i, j, k-1) + thbar2)) / ( dz ) );

    // compute and save to the array
    buf0 = yvort_solenoid; 
    BUF(i, j, k) = -cp*(dthdz*dpidx - dthdx*dpidz); 
}

__device__ void calc_zvort_solenoid(float *pipert, float *thrhopert, float *zvort_solenoid, \
                                    float dx, float dy, int i, int j, int k, int NX, int NY) {
    float *buf0 = pipert;
    float dpidx = ( (BUF(i+1, j, k) - BUF(i-1, j, k)) / ( 2*dx ) );
    float dpidy = ( (BUF(i, j+1, k) - BUF(i, j-1, k)) / ( 2*dy ) );

    buf0 = thrhopert;
    float dthdx = ( (BUF(i+1, j, k) - BUF(i-1, j, k)) / ( 2*dx ) );
    float dthdy = ( (BUF(i, j+1, k) - BUF(i, j-1, k)) / ( 2*dy ) );

    // compute and save to the array
    buf0 = zvort_solenoid; 
    BUF(i, j, k) = -cp*(dthdx*dpidy - dthdy*dpidx); 
}

#endif
