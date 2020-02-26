#include <iostream>
#include <stdio.h>
#include "../include/datastructs.h"
#include "../include/macros.h"
#ifndef TURB_CALC
#define TURB_CALC

__device__ void calcrf(float *rhopert, float *rho0, float *rhof, int i, int j, int k, int NX, int NY) {
    // use the w staggered grid
    float *wstag = rhof;
    float *buf0 = rhopert;

    if (k >= 1) {
        WA(i, j, k) =  0.5*( (BUF(i, j, k-1) + rho0[k-1]) + (BUF(i, j, k) + rho0[k]) );
    }

	// extrapolate for the lowest point
    else if (k == 0) {
        WA(i, j, 0) = (1.75*(BUF(i, j, 0) + rho0[0]) - (BUF(i, j, 1) + rho0[1]) + 0.25*(BUF(i, j, 2) + rho0[2]));
    }
}

// Calculating the strain rate terms requires more than one stencil operation to get it right.
// This first one handles the calculation of divergence and "easily" calculable vertical terms,
// ie s11, s22, s33, and s12. The values of s13 and s23 are set in calcstrain2, and surface boundary
// conditions are set in gettau. 
__device__ void calcstrain1(float *ustag, float *vstag, float *wstag, float *rhopert, float *rho0, \
		                float *s11, float *s12, float *s22, float *s33, float dx, float dy, float dz, \
						int i, int j, int k, int NX, int NY) {
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
				 *0.25*( (rho1+rho2+rho3+rho4) );
}

// This routine computes the remaining strain rate ternsors, s13 and s23. These have to go on a different
// kernel call because the stencils require handling the vertical loops differently, meaning they cannot
// be combined into a single kernel call. Breaking them up into individual kernels for each stress
// tensor would likely not give enough work per thread either. 
__device__ void calcstrain2(float *ustag, float *vstag, float *wstag, float *rhof, float *s13, float *s23, \
		                 float dx, float dy, float dz, int i, int j, int k, int NX, int NY) {
	float *buf0 = rhof;
	float rf1 = BUF(i, j, k);
	float rf2 = BUF(i-1, j, k);
	float rf3 = BUF(i, j-1, k);

	buf0 = s13;
	BUF(i, j, k) = 0.5*( (WA(i, j, k) - WA(i-1, j, k))*(1./dx) \
					   + (UA(i, j, k) - UA(i, j, k-1))*(1./dz) ) \
				   *0.5*(rf1 + rf2);
	
	buf0 = s23;
	BUF(i, j, k) = 0.5*( (WA(i, j, k) - WA(i, j-1, k))*(1./dy) \
					   + (VA(i, j, k) - VA(i, j, k-1))*(1./dz) ) \
				   *0.5*(rf1 + rf3);
}


// Similar to the strain kernels, we need two of these for computing some of the vertical components due to the 
// way the vertical stencils work with boundary conditions. This kernel also probably doesn't know whether z(k==0) 
// is the actual surface or the lowest level of an array defined above the surface.
__device__ void gettau1(float *km, float *t11, float *t12, float *t22, float *t33, int i, int j, int k, int NX, int NY) {
	float *buf0 = km;
	// KM is defined on W points - get on scalar vertical points
	float kmval = 0.5*(BUF(i, j, k) + BUF(i, j, k+1));
	
	// These arrays already have rho * S_ij in them!
	buf0 = t11;
	BUF(i, j, k) = 2.0 * kmval * BUF(i, j, k);
	
	buf0 = t22;
	BUF(i, j, k) = 2.0 * kmval * BUF(i, j, k);

	buf0 = t33;
	BUF(i, j, k) = 2.0 * kmval * BUF(i, j, k);

	buf0 = km;
	kmval = 0.125 * ( ( (BUF(i-1,j-1,k  )+BUF(i,j,k  ))+(BUF(i-1,j,k  )+BUF(i,j-1,k  )) )   \
                   +  ( (BUF(i-1,j-1,k+1)+BUF(i,j,k+1))+(BUF(i-1,j,k+1)+BUF(i,j-1,k+1)) ) );
	buf0 = t12;
	BUF(i, j, k) = 2.0 * kmval * BUF(i, j, k);
}

__device__ void gettau2(float *km, float *t13, float *t23, int i, int j, int k, int NX, int NY) {
	float *buf0 = km;
	float kmval1 = 0.5 * (BUF(i, j, k) + BUF(i-1, j, k)); // km on u points
	float kmval2 = 0.5 * (BUF(i, j, k) + BUF(i, j-1, k)); // km on v points

	buf0 = t13;
	BUF(i, j, k) = 2.0 * kmval1 * BUF(i, j, k);

	buf0 = t23;
	BUF(i, j, k) = 2.0 * kmval2 * BUF(i, j, k);
}

__device__ void calc_turbu(float *t11, float *t12, float *t13, float *rhopert, float *rho0, float *turbu, \
		                   float dx, float dy, float dz, int i, int j, int k, int NX, int NY) {
    float *ustag, *buf0, *dum0;

    // tau 11
    dum0 = t11;
    float turbx = ((TEM(i, j, k) - TEM(i-1, j, k)) / dx);

    // tau 12
    dum0 = t12;
    float turby = ((TEM(i, j+1, k) - TEM(i, j, k)) / dy);

    // tau 13
    dum0 = t13;
    float turbz = ((TEM(i, j, k+1) - TEM(i, j, k)) / dz);

    buf0 = rhopert;
    // calculate the momentum tendency now
    float rru0 = 1.0 / (0.5 * ((BUF(i-1, j, k) + rho0[k]) + (BUF(i, j, k) + rho0[k])));
    ustag = turbu;
    UA(i, j, k) = ( turbx + turby + turbz ) * rru0; 
}

__device__ void calc_turbv(float *t12, float *t22, float *t23, float *rhopert, float *rho0, float *turbv, \
		                   float dx, float dy, float dz, int i, int j, int k, int NX, int NY) {
    float *vstag, *buf0, *dum0;

    // tau 12
    dum0 = t12;
    float turbx = ((TEM(i+1, j, k) - TEM(i, j, k)) / dx);

    // tau 22
    dum0 = t22;
    float turby = ((TEM(i, j, k) - TEM(i, j-1, k)) / dy);

    // tau 23
    dum0 = t23;
    float turbz = ((TEM(i, j, k+1) - TEM(i, j, k)) / dz);

    buf0 = rhopert;
    // calculate the momentum tendency now
    float rrv0 = 1.0 / (0.5 * ((BUF(i, j-1, k) + rho0[k]) + (BUF(i, j, k) + rho0[k])));
    vstag = turbv;
    VA(i, j, k) = ( turbx + turby + turbz ) * rrv0; 
}

__device__ void calc_turbw(float *t13, float *t23, float *t33, float *rhof, float *turbw, \
		                   float dx, float dy, float dz, int i, int j, int k, int NX, int NY) {
    float *wstag, *buf0, *dum0;

    // tau 13
    dum0 = t13;
    float turbx = ((TEM(i+1, j, k) - TEM(i, j, k)) / dx);

    // tau 23
    dum0 = t23;
    float turby = ((TEM(i, j+1, k) - TEM(i, j, k)) / dy);

    // tau 33
    dum0 = t33;
    float turbz = ((TEM(i, j, k) - TEM(i, j, k-1)) / dz);

    buf0 = rhof;
    // calculate the momentum tendency now
    float rrf = 1.0 / BUF(i, j, k);
    wstag = turbw;
    WA(i, j, k) = ( turbx + turby + turbz ) * rrf; 
}

#endif
