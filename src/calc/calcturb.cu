#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#ifndef TURB_CALC
#define TURB_CALC

__device__ void calcrf(float *rhopert, float *rho0, float *rhof, int i, int j, int k, int NX, int NY) {
    // use the w staggered grid
    float *wstag = data->rhof;
    float *buf0 = data->rhopert;

    if (k >= 2) {
        WA(i, j, k) =  0.5*( (BUF(i, j, k-1) + rho0[k-1]) + (BUF(i, j, k) + rho0[k]) );
    }

	// extrapolate for the lowest point
    else if (k == 0) {
        WA(i, j, 0) = (1.75*(BUF(i, j, 1) + rho0[1]) - (BUF(i, j, 2) + rho0[2]) + 0.25*(BUF(i, j, 3) + rho0[3]));
    }
}

// Calculating the strain rate terms requires more than one stencil operation to get it right.
// This first one handles the calculation of divergence and "easily" calculable vertical terms,
// ie s11, s22, s33, and s12. The values of s13 and s23 are set in calcstrain2, and surface boundary
// conditions are set in gettau. 
__device__ void calcstrain1(float *ustag, float *vstag, float *wstag, float *rhopert, float *rho0, \
		                float *s11, float *s12, float *s22, float *s33, float dx, float dy, float dz, \
						int i, int j, int k, int nx, int ny) {
	float *buf0 = rhopert;
	float rho1 = BUF(i, j, k) + rho0[k];
	float rho2 = BUF(i-1, j-1, k) + rho0[k];
	float rho3 = BUF(i-1, j, k) + rho0[k];
	float rho4 = BUF(i, j-1, k) + rho0[k];

	// Strain rate tensor 11
	buf0 = s11;
	BUF(i, j, k) = rho1 * (UA(i+1, j, k) - UA(i, j, k)) * (1./dx);

	// Strain rate tensor 22
	buf0 = s22;
	BUF(i, j, k) = rho1 * (VA(i, j+1, k) - VA(i, j, k)) * (1./dy);

	// Strain rate tensor 33
	buf0 = s33;
	BUF(i, j, k) = rho1 * (WA(i, j, k+1) - WA(i, j, k)) * (1./dz);

	// Strain rate tensor 12
	buf0 = s12;
	BUF(i, j, k) = 0.5*( (UA(i, j, k) - UA(i, j-1, k))*(1./dy) \
			          +  (VA(i, j, k) - VA(i-1, j, k))*(1./dx) ) \
				 *0.25*( (rho+rho2+rho3+rho3) );
}

// This routine computes the remaining strain rate ternsors, s13 and s23. These have to go on a different
// kernel call because the stencils require handling the vertical loops differently, meaning they cannot
// be combined into a single kernel call. Breaking them up into individual kernels for each stress
// tensor would likely not give enough work per thread either. 
__device__ void calcstrain2(float *ustag, float *vstag, float *wstag, float *rhof, float *s13, float *s23, \
		                 float dx, float dy, float dz, int i, int j, int k, int nx, int ny) {
	float *buf0 = rhof;
	rf1 = BUF(i, j, k);
	rf2 = BUF(i-1, j, k);
	rf3 = BUF(i, j-1, k)

	buf0 = s13;
	BUF(i, j, k) = 0.5*( (WA(i, j, k) - WA(i-1, j, k))*(1./dx) \
					   + (UA(i, j, k) - UA(i, j, k-1))*(1./dz) ) \
				   *0.5*(rf1 + rf2);
	
	buf0 = s23;
	BUF(i, j, k) = 0.5*( (WA(i, j, k) - WA(i, j-1, k))*(1./dy) \
					   + (VA(i, j, k) - VA(i, j, k-1))*(1./dz) ) \
				   *0.5*(rf1 + rf3);
}


__device__ void gettau(datagrid *grid, model_data *data, int *idx_4D, int NX, int NY, int NZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float *dum0, *buf0, *wstag, *kmstag;

    kmstag = data->kmh;
    buf0 = data->rhopert;

    // NOTE: Base state arrays have a different grid index to them because there is no ghost zone.
    // For example, rho0[0] corresponds to zh[1]. We need to be careful and make sure we offset 
    // our indices appropriately

    // tau 11
    dum0 = data->tem1;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k]);
    // tau 22
    dum0 = data->tem3;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k]);
    // tau 33
    dum0 = data->tem4;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * (KM4D(i, j, k, t) + KM4D(i, j, k+1, t))*(BUF4D(i, j, k, t) + grid->rho0[k]);

    // tau 12
    dum0 = data->tem2;
    TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.03125 * \
                        ( ( ( KM4D(i-1, j-1, k, t) + KM4D(i, j, k, t) ) + ( KM4D(i-1, j, k, t) + KM4D(i, j-1, k, t) ) ) \
                         +( ( KM4D(i-1, j-1, k+1, t) + KM4D(i, j, k+1, t) ) + ( KM4D(i-1, j, k+1, t) + KM4D(i, j-1, k+1, t) ) ) ) \
                         *( ( (BUF4D(i-1, j-1, k, t) + grid->rho0[k]) + (BUF4D(i, j, k, t) + grid->rho0[k]) ) \
                           + ((BUF4D(i-1, j, k, t) + grid->rho0[k]) + (BUF4D(i, j-1, k, t) + grid->rho0[k]) ) );
    if (k == 1) {
        // we'll go ahead and apply the zero strain condition on the lower boundary/ghost zone
        // for tau 13 and tau 23
        // tau 13 boundary
        dum0 = data->tem5;
        TEM4D(i, j, 0, t) = 0.0;
        // tau 23 boundary
        dum0 = data->tem6;
        TEM4D(i, j, 0, t) = 0.0;
    }

    if ((k >= 1)) {
        // tau 13
        dum0 = data->tem5;
        wstag = data->rhof; // rather than make a new maro, we'll just use the WA4D macro
        TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.25 \
                                *( KM4D(i-1, j, k, t) + KM4D(i, j, k, t) ) \
                                *( (WA4D(i-1, j, k, t)) + (WA4D(i, j, k, t)) ); 
        // tau 23
        dum0 = data->tem6;
        TEM4D(i, j, k, t) = TEM4D(i, j, k, t) * 0.25 \
                                *( KM4D(i, j-1, k, t) + KM4D(i, j, k, t) ) \
                                *( (WA4D(i, j-1, k, t) + WA4D(i, j, k, t)) ); 
    }
}

// TO-DO: These next 3 kernels effectively do the same thing... This is a place where consolidation needs to take place
// for both readability and maintainability. While reducing the duplicate kernels in one part of this, looking into
// the CM1r19.8 formulation could help since it does operations in a slightly different way than CM1r16, which this
// was primarily based on. 
__device__ void calc_turbu(float *xturb, float *yturb, float *zturb, float *rhopert, float *rho0, float *turbu, \
		                   float dx, float dy, float dz, int i, int j, int k, int NX, int NY) {
    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *ustag, *buf0, *dum0;
	/*
     *float dx = xf(i) - xf(i-1);
     *float dy = yf(j) - yf(j-1);
     *float dz = zf(k+1) - zf(k);
	 */

    // tau 11
    dum0 = xturb;
    float turbx = ((TEM(i, j, k) - TEM(i-1, j, k)) / dx);

    // tau 12
    dum0 = yturb;
    float turby = ((TEM(i, j+1, k) - TEM(i, j, k)) / dy);

    // tau 13
    dum0 = zturb;
    float turbz = ((TEM(i, j, k+1) - TEM4D(i, j, k)) / dz);

    buf0 = rhopert;
    // calculate the momentum tendency now
    float rru0 = 1.0 / (0.5 * ((BUF(i-1, j, k) + rho0[k]) + (BUF(i, j, k) + rho0[k])));
    ustag = turbu;
    UA(i, j, k) = ( turbx + turby + turbz ) * rru0; 
}

__device__ void calc_turbv(float *xturb, float *yturb, float *zturb, float *rhopert, float *rho0, float *turbv, \
		                   float dx, float dy, float dz, int i, int j, int k, int NX, int NY) {
    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *vstag, *buf0, *dum0;
	/*
     *float dx = xf(i) - xf(i-1);
     *float dy = yf(j) - yf(j-1);
     *float dz = zf(k+1) - zf(k);
	 */

    // tau 12
    dum0 = xturb;
    float turbx = ((TEM(i+1, j, k) - TEM(i, j, k)) / dx);

    // tau 22
    dum0 = yturb;
    float turby = ((TEM(i, j, k) - TEM(i, j-1, k)) / dy);

    // tau 23
    dum0 = zturb;
    float turbz = ((TEM(i, j, k+1) - TEM(i, j, k)) / dz);

    buf0 = rhopert;
    // calculate the momentum tendency now
    float rrv0 = 1.0 / (0.5 * ((BUF(i, j-1, k) + rho0[k]) + (BUF(i, j, k) + rho0[k])));
    vstag = turbv;
    VA(i, j, k) = ( turbx + turby + turbz ) * rrv0; 
}

__device__ void calc_turbw(float *xturb, float *yturb, float *zturb, float *rhopert, float *rho0, float *turbw, \
		                   float dx, float dy, float dz, int i, int j, int k, int NX, int NY) {
    // the momentum tendencies will lie on their staggered mesh counterparts,
    // so we will use the stagger macros to store the data in order to maintain
    // consistency
    float *wstag, *buf0, *dum0;
	/*
     *float dx = xf(i) - xf(i-1);
     *float dy = yf(j) - yf(j-1);
     *float dz = zf(k+1) - zf(k);
	 */

    // tau 13
    dum0 = xturb;
    float turbx = ((TEM(i+1, j, k) - TEM(i, j, k)) / dx);

    // tau 23
    dum0 = yturb;
    float turby = ((TEM(i, j+1, k) - TEM(i, j, k)) / dy);

    // tau 33
    dum0 = zturb;
    float turbz = ((TEM(i, j, k) - TEM(i, j, k-1)) / dz);

    buf0 = rhopert;
    // calculate the momentum tendency now
    float rrf = 1.0 / (0.5 * (BUF(i, j, k-1) + rho0[k-1] + BUF(i, j, k) + rho0[k]));
    wstag = turbw;
    WA(i, j, k) = ( turbx + turby + turbz ) * rrf; 
}
#endif
