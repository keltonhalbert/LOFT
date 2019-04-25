#include <iostream>
#include <stdio.h>
#include "datastructs.cpp"
#include "interp.cu"
#ifndef INTEGRATE_CU
#define INTEGRATE_CU

using namespace std;
#define P2(t,p,mt) (((p)*(mt))+(t))
// this is an error checking helper function for processes
// that run on the GPU. Without calling this, the GPU can
// fail to execute but the program won't crash or report it.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      cout << cudaGetErrorString(code) << endl;
      if (abort) exit(code);
   }
}


// rf is the rho field on the vertical staggered mesh. George does this for turbulence closure
// since K terms and derivatives are all done on the W mesh. This is calculated by doing an extrapolation 
// from the scalar mesh to the W staggered mesh. As described below, c1 and c2 are distances of the staggered mesh from
// the scalar mesh normalized by the grid spacing dz, and since in an isotropic mesh the stagger is exactly half way
// between scalar points, is 0.5. This is hard coded, but even in our stretch zone above 10km, it's really close
// to 0.5. This is probably only violated for wildly stretched meshes, which we don't use because they're dumb.
// Mostly just noting this for future reference, because this will need correcting for stretched meshes.
__device__ void calcrf(datagrid grid, float *rho, float *rf, int *idx_4D, int MX, int MY, int MZ) {

    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    // c1 and c2 are both 0.5 for isotropic staggered meshes. We are hard coding this to be the case here for our data,
    // but is not necessarily true for all simulations.
    float c1 = 0.5; float c2 = 0.5;
    if ( k == 0) {
        float rho1 = grid.rho0[k] + rho[arrayIndex(i, j, 1, t, MX, MY, MZ)]; 
        float rho2 = grid.rho0[k] + rho[arrayIndex(i, j, 2, t, MX, MY, MZ)];
        float rho3 = grid.rho0[k] + rho[arrayIndex(i, j, 3, t, MX, MY, MZ)];
        rf[arrayIndex(i, j, 0, t, MX, MY, MZ)] = (1.75*rho1-rho2+0.25*rho3);
    }
    else {

        float rho1 = grid.rho0[k] + rho[arrayIndex(i, j, k, t, MX, MY, MZ)]; 
        float rho2 = grid.rho0[k] + rho[arrayIndex(i, j, k+1, t, MX, MY, MZ)];
        rf[arrayIndex(i, j, k, t, MX, MY, MZ)] = ( c1*rho1 + c2*rho2);
    }
    // there's technically a top boundary condition in CM1, but we're ignoring because we hope to be far away from the upper boundary.
}

// calculate the deformation terms for the turbulence diagnostics. They get stored in the 
// arrays later designated for tau stress tensors and variables are named according to
// tensor notation
__device__ void calcdef(datagrid grid, float *uarr, float *varr, float *warr, float *rho, float *rf,\
                        float *s11, float *s12, float *s13, float *s22, float *s23, float *s33, \
                        int *idx_4D, int MX, int MY, int MZ) {

    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xf[i+1] - grid.xf[i];

    // tau 11. Derivative is du/dx therefore use the staggered mesh and forward difference it to get it on the scalar mesh
    float rho1 = grid.rho0[k] + rho[arrayIndex(i, j, k, t, MX, MY, MZ)];
    s11[arrayIndex(i, j, k, t, MX, MY, MZ)] = rho1*( uarr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dx;


    // tau 12. Derivatives are no longer on the staggered meshes since it's du/dy and dv/dx. Therefore, an averaging step
    // will be required later at some point.
    float dy = grid.yh[j+1] - grid.yh[j];
    dx = grid.xh[i+1] - grid.xh[i];
    float rho2 = grid.rho0[k] + rho[arrayIndex(i+1, j+1, k, t, MX, MY, MZ)];
    float rho3 = grid.rho0[k] + rho[arrayIndex(i, j+1, k, t, MX, MY, MZ)];
    float rho4 = grid.rho0[k] + rho[arrayIndex(i+1, j, k, t, MX, MY, MZ)];
    s12[arrayIndex(i, j, k, t, MX, MY, MZ)] = 0.5 * ( ( uarr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dy \
                                            + ( varr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dx ) \
                                            * 0.25*( (rho1 + rho2) + (rho3 + rho4) );

   // tau 22. Derivative is dv/dy therefore use the staggered mesh and forward difference it to get it on the scalar mesh
   dy = grid.yf[j+1] - grid.yf[j];
   s22[arrayIndex(i, j, k, t, MX, MY, MZ)] = rho1 * ( varr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dy;


   // tau 33. Derivative is du/dz and therefore use the staggered mesh and forward difference it to get it on the scalar mesh
   float dz = grid.zf[k+1] - grid.zf[k];
   s33[arrayIndex(i, j, k, t, MX, MY, MZ)] = rho1 * ( warr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - warr[arrayIndex(i, k, k, t, MX, MY, MZ)] ) / dz;

   // data above the lower boundary
   if ( k >= 1.) {

       // tau 13. Derivative is no longer on staggered mesh and will require an average to correct the data to the scalar mesh
       dx = grid.xh[i+1] - grid.xh[i];
       dz = grid.zh[k+1] - grid.zh[k];
       s13[arrayIndex(i, j, k, t, MX, MY, MZ)] = 0.5*( ( warr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - warr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dx \
                                               + ( uarr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dz ) \
                                                 *0.5*( rf[arrayIndex(i, j, k, t, MX, MY, MZ)] + rf[arrayIndex(i+1, j, k, t, MX, MY, MZ)]);

       // tau 23. Derivative is no longer on staggered mesh and will require an average to correct the data to the scalar mesh
       dy = grid.yh[j+1] - grid.yh[j];
       s23[arrayIndex(i, j, k, t, MX, MY, MZ)] = 0.5*( ( warr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - warr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dy \
                                               + ( varr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)] ) / dz) \
                                               *0.5*( rf[arrayIndex(i, j, k, t, MX, MY, MZ)]+rf[arrayIndex(i, j+1, k, t, MX, MY, MZ)] );

   }

   // lower boundary condition
   else {
       s13[arrayIndex(i, j, k, t, MX, MY, MZ)] = 0.0;
       s23[arrayIndex(i, j, k, t, MX, MY, MZ)] = 0.0;
   }
}

// take the output from calcdef and compute the full stress tensor tau
// !NOTE: turb coefficients are defined on w points
__device__ void gettau(datagrid grid, float *rho, float *khh, \
                        float *t11, float *t12, float *t13, float *t22, float *t23, float *t33, \
                        int *idx_4D, int MX, int MY, int MZ) {

    // get the i,j,k,t index of where we are doing
    // our differencing
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];


    // kmh = Pr * khh and Pr is 1./3.
    float tem = (1./3.)*khh[arrayIndex(i, k, k, t, MX, MY, MZ)] + (1./3.)*khh[arrayIndex(i, j, k+1, t, MX, MY, MZ)];


    // these are conveniently on points we know... but that convecience will end shortly
    t11[arrayIndex(i, j, k, t, MX, MY, MZ)] = t11[arrayIndex(i, j, k, t, MX, MY, MZ)] * tem;

    t22[arrayIndex(i, j, k, t, MX, MY, MZ)] = t22[arrayIndex(i, j, k, t, MX, MY, MZ)] * tem;

    t33[arrayIndex(i, j, k, t, MX, MY, MZ)] = t33[arrayIndex(i, j, k, t, MX, MY, MZ)] * tem;

    // do some 8 point averaging of our kmh in space
    float kmh1 = khh[arrayIndex(i-1, j-1, k, t, MX, MY, MZ )] * 1./3.;
    float kmh2 = khh[arrayIndex(i, j, k, t, MX, MY, MZ )] * 1./3.;
    float kmh3 = khh[arrayIndex(i-1, j, k, t, MX, MY, MZ )] * 1./3.;
    float kmh4 = khh[arrayIndex(i, j-1, k, t, MX, MY, MZ  )] * 1./3.;
    float kmh5 = khh[arrayIndex(i-1, j-1, k+1, t, MX, MY, MZ )] * 1./3.;
    float kmh6 = khh[arrayIndex(i, j, k+1, t, MX, MY, MZ )] * 1./3.;
    float kmh7 = khh[arrayIndex(i-1, j, k+1, t, MX, MY, MZ )] * 1./3.;
    float kmh8 = khh[arrayIndex(i,j-1,k+1, t, MX, MY, MZ)] * 1./3.;
    t12[arrayIndex(i, j, k, t, MX, MY, MZ)] = t12[arrayIndex(i, j, k, t, MX, MY, MZ)]*.025 \
    *( ( (kmh1+kmh2)+(kmh3+kmh4) )+( (kmh5+kmh6)+(kmh7+kmh8) ) );

    if (k >= 1) {
        float kmv1 = kmh2; 
        float kmv2 = khh[arrayIndex(i+1, j, k, t, MX, MY, MZ)] * 1./3.;
        float kmv3 = khh[arrayIndex(i, j+1, k, t, MX, MY, MZ)] * 1./3.;
        t13[arrayIndex(i, j, k, t, MX, MY, MZ)] = t13[arrayIndex(i, j, k, t, MX, MY, MZ)] * ( kmv1 + kmv2 );
        t23[arrayIndex(i, j, k, t, MX, MY, MZ)] = t23[arrayIndex(i, j, k, t, MX, MY, MZ)] * ( kmv1 + kmv3 ); 

    }
    else {

        // lower boundary condition
        t13[arrayIndex(i, j, k, t, MX, MY, MZ)] = 0.0;
        t23[arrayIndex(i, j, k, t, MX, MY, MZ)] = 0.0;

    }
    
}

// NOTE: Care has been taken to best preserve the numerical accuracy of how things are done in CM1, 
// but some areas where George does a backwards difference have been replaced with a forward difference.
// It is unknown to me at this time how much this will impact the overall result, but it could result in
// a pesky "off by one error" of sorts where in cartesian space, the values are slightly off. One day
// I'll probably revisit this and refactor it to make it 100% accurate with CM1, but Fortran's inconsistency
// with 0 indexed vs 1 indexed arrays confuse me sometimes, and we're not dealing with ghost zones

// calculate the turbulence term of the momentum equation for U. Eventually this will be modified to be
// the turbulence term for a component of vorticity (I think). 
__device__ void calc_turbu(datagrid grid, float *t11, float *t12, float *t13, float *turbx, float *turby, float *turbz, \
                        int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xh[i+1] - grid.xh[i];
    float dy = grid.yh[j+1] - grid.yh[j];
    float dz = grid.zh[k+1] - grid.zh[k];
    turbx[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t11[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - t11[arrayIndex(i, j, k, t, MX, MY, MZ)])/dx;
    turby[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t12[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - t12[arrayIndex(i, j, k, t, MX, MY, MZ)])/dy;
    turbz[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t13[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - t13[arrayIndex(i, j, k, t, MX, MY, MZ)])/dz;

}


// calculate the turbulence term of the momentum equation for V. Eventually this will be modified to be
// the turbulence term for a component of vorticity (I think). 
__device__ void calc_turbv(datagrid grid, float *t12, float *t22, float *t23, float *turbx, float *turby, float *turbz,
                        int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xh[i+1] - grid.xh[i];
    float dy = grid.yh[j+1] - grid.yh[j];
    float dz = grid.zh[k+1] - grid.zh[k];
    turbx[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t12[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - t12[arrayIndex(i, j, k, t, MX, MY, MZ)]) / dx;
    turby[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t22[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - t22[arrayIndex(i, j, k, t, MX, MY, MZ)]) / dy;
    turbz[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t23[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - t23[arrayIndex(i, j, k, t, MX, MY, MZ)]) / dz;

}

// calculate the turbulence term of the momentum equation for W. Eventually this will be modified to be
// the turbulence term for a component of vorticity (I think). 
__device__ void calc_turbw(datagrid grid, float *t13, float *t23, float *t33, float *turbx, float *turby, float *turbz,
                        int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xh[i+1] - grid.xh[i];
    float dy = grid.yh[j+1] - grid.yh[j];
    float dz = grid.zh[k+1] - grid.zh[k];
    turbx[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t13[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - t13[arrayIndex(i, j, k, t, MX, MY, MZ)]) / dx;
    turby[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t23[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - t23[arrayIndex(i, j, k, t, MX, MY, MZ)]) / dy;
    turbz[arrayIndex(i, j, k, t, MX, MY, MZ)] = (t33[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - t33[arrayIndex(i, j, k, t, MX, MY, MZ)]) / dz;

}

__device__ void calc_xvort(datagrid grid, float *warr, float *varr, float *xvort, int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dz = grid.zh[k+1] - grid.zh[k];
    float dy = grid.yh[j+1] - grid.yh[j];
    float dw = warr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - warr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float dv = varr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    xvort[arrayIndex(i, j, k, t, MX, MY, MZ)] = ( dw / dy ) - ( dv / dz);

}

__device__ void calc_yvort(datagrid grid, float *uarr, float *warr, float *yvort, int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dz = grid.zh[k+1] - grid.zh[k];
    float dx = grid.xh[i+1] - grid.xh[i];
    float dw = warr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - warr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float du = uarr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    yvort[arrayIndex(i, j, k, t, MX, MY, MZ)] = ( du / dz ) - ( dw / dx);

}

__device__ void calc_zvort(datagrid grid, float *uarr, float *varr, float *zvort, int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xh[i+1] - grid.xh[i];
    float dy = grid.yh[j+1] - grid.yh[j];
    float dv = varr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float du = uarr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    zvort[arrayIndex(i, j, k, t, MX, MY, MZ)] = ( dv / dx ) - ( du / dy);
    //printf("%f, %i, %i, %i, %i\n", zvort[arrayIndex(i, j, k, t, MX, MY, MZ)], i, j, k, t);
}
// tilting terms: do center difference, not forward
__device__ void calc_zvorttilt(datagrid grid, float *zvorttilt, float *xvort, float *yvort, float*warr, int *idx_4D, int MX, int MY, int MZ) {
	int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xh[i+1] - grid.xh[i-1];
    float dy = grid.yh[j+1] - grid.yh[j-1];
    float dwdx = (warr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - warr[arrayIndex(i-1, j, k, t, MX, MY, MZ)]) / dx;
    float dwdy = (warr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - warr[arrayIndex(i, j-1, k, t, MX, MY, MZ)]) / dy;
    float xv = xvort[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float yv = yvort[arrayIndex(i, j, k, t, MX, MY, MZ)];

    zvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)] = xv*dwdx + yv*dwdy;
    if (i == 1) zvorttilt[arrayIndex(0, j, k, t, MX, MY, MZ)] = zvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)];
    if (j == 1) zvorttilt[arrayIndex(i, 0, k, t, MX, MY, MZ)] = zvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)];

}

__device__ void calc_xvorttilt(datagrid grid, float *xvorttilt, float *yvort, float *zvort, float*uarr, int *idx_4D, int MX, int MY, int MZ) {
	int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dy = grid.yh[j+1] - grid.yh[j-1];
    float dz = grid.zh[k+1] - grid.zh[k-1];
    float dudy = (uarr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - uarr[arrayIndex(i, j-1, k, t, MX, MY, MZ)]) / dy;
    float dudz = (uarr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k-1, t, MX, MY, MZ)]) / dz;
    float yv = yvort[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float zv = zvort[arrayIndex(i, j, k, t, MX, MY, MZ)];

    xvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)] = yv*dudy + zv*dudz;
    if (j == 1) xvorttilt[arrayIndex(i, 0, k, t, MX, MY, MZ)] = xvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)];
    if (k == 1) xvorttilt[arrayIndex(i, j, 0, t, MX, MY, MZ)] = xvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)];

}

__device__ void calc_yvorttilt(datagrid grid, float *yvorttilt, float *xvort, float *zvort, float*varr, int *idx_4D, int MX, int MY, int MZ) {
	int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xh[i+1] - grid.xh[i-1];
    float dz = grid.zh[k+1] - grid.zh[k-1];
    float dvdx = (varr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - varr[arrayIndex(i-1, j, k, t, MX, MY, MZ)]) / dx;
    float dvdz = (varr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k-1, t, MX, MY, MZ)]) / dz;
    float xv = xvort[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float zv = zvort[arrayIndex(i, j, k, t, MX, MY, MZ)];

    yvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)] = xv*dvdx + zv*dvdz;
    if (i == 1) yvorttilt[arrayIndex(0, j, k, t, MX, MY, MZ)] = yvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)];
    if (k == 1) yvorttilt[arrayIndex(i, j, 0, t, MX, MY, MZ)] = yvorttilt[arrayIndex(i, j, k, t, MX, MY, MZ)];

}

/* calculate the stretching term for vertical vorticity. The stretching functions naturally result on
scalar grid points, nothing needs to be done after this*/
__device__ void calc_zvortstretch(datagrid grid, float *zvortstretch, float *zvort, float *uarr, float *varr, int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xf[i+1] - grid.xf[i];
    float dy = grid.yf[j+1] - grid.yf[j];
    float du = uarr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float dv = varr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float zv = zvort[arrayIndex(i, j, k, t, MX, MY, MZ)];

    // du/dx and dv/dy end up on the scalar grid, and zvort is already averaged to the
    // scalar grid, so we're done here.
    zvortstretch[arrayIndex(i, j, k, t, MX, MY, MZ)] = -1.*zv*( (du/dx) + (dv/dy) );

}

/* calculate the stretching term for vertical vorticity. The stretching functions*/
__device__ void calc_yvortstretch(datagrid grid, float *yvortstretch, float *yvort, float *uarr, float *warr, int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float dx = grid.xf[i+1] - grid.xf[i];
    float dz = grid.zf[k+1] - grid.zf[k];
    float du = uarr[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - uarr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float dw = warr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - warr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float yv = yvort[arrayIndex(i, j, k, t, MX, MY, MZ)];

    // du/dx and dv/dy end up on the scalar grid, and zvort is already averaged to the
    // scalar grid, so we're done here.
    yvortstretch[arrayIndex(i, j, k, t, MX, MY, MZ)] = -1.*yv*( (du/dx) + (dw/dz) );

}

/* calculate the stretching term for vertical vorticity. The stretching functions*/
__device__ void calc_xvortstretch(datagrid grid, float *xvortstretch, float *xvort, float *varr, float *warr, int *idx_4D, int MX, int MY, int MZ) {
    int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];
    
    //printf("%i %i %i %i\n", i, j, k ,t);
   	float dz = grid.zf[k+1] - grid.zf[k];
    float dy = grid.yf[j+1] - grid.yf[j];

    


    float dv = varr[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - varr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float dw = warr[arrayIndex(i, j, k+1, t, MX, MY, MZ)] - warr[arrayIndex(i, j, k, t, MX, MY, MZ)];
    float xv = xvort[arrayIndex(i, j, k, t, MX, MY, MZ)];


    // du/dx and dv/dy end up on the scalar grid, and zvort is already averaged to the
    // scalar grid, so we're done here.
    xvortstretch[arrayIndex(i, j, k, t, MX, MY, MZ)] =  -1.*xv*( (dv/dy) + (dw/dz) );

}


__device__ void calc_xvortbaro(datagrid grid, float *xbaro, float *thrhopert, int *idx_4D, int MX, int MY, int MZ) {

	int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float reps = 461.5/287.04;
    float coeff = 9.81 / (grid.th0[k] * (1.0 + reps*grid.qv0[k]/(1.0+grid.qv0[k])));
    float dy = grid.yh[j+1] - grid.yh[j-1];

    xbaro[arrayIndex(i, j, k, t, MX, MY, MZ)] = coeff * (thrhopert[arrayIndex(i, j+1, k, t, MX, MY, MZ)] - thrhopert[arrayIndex(i, j-1, k, t, MX, MY, MZ)]) / dy;

}

__device__ void calc_yvortbaro(datagrid grid, float *ybaro, float *thrhopert, int *idx_4D, int MX, int MY, int MZ) {

	int i = idx_4D[0];
    int j = idx_4D[1];
    int k = idx_4D[2];
    int t = idx_4D[3];

    float reps = 461.5/287.04;
    float coeff = 9.81 / (grid.th0[k] * (1.0 + reps*grid.qv0[k]/(1.0+grid.qv0[k])));
    float dx = grid.xh[i+1] - grid.xh[i-1];

    ybaro[arrayIndex(i, j, k, t, MX, MY, MZ)] = -1.*coeff * (thrhopert[arrayIndex(i+1, j, k, t, MX, MY, MZ)] - thrhopert[arrayIndex(i-1, j, k, t, MX, MY, MZ)]) / dx;

}


__global__ void doCalcrf(datagrid grid, float *rho_time_chunk, float *rhof, int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calcrf(grid, rho_time_chunk, rhof, idx_4D, MX, MY, MZ);
        }
    }
}

__global__ void doCalcdef(datagrid grid, float *uarr, float *varr, float *warr, float *rho, float *rf, \
                        float *s11, float *s12, float *s13, float *s22, float *s23, float *s33, \
                        int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calcdef(grid, uarr, varr, warr, rho, rf, s11, s12, s13, s22, s23, s33, idx_4D, MX, MY, MZ);
        }
    }
}

/*
__global__ void doGettau() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
        }
    }
}

__global__ void goTurbu() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
        }
    }
}

__global__ void doTurbv() { 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
        }
    }
}

__global__ void doTurbw() { 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
        }
    }
}
*/

/* NOTE FOR VORTICITY KERNELS!!!!!!
Our subset doesn't have ghost zones per-say, but we assume
that our parcels aren't on the borders of the arrays. This is
partly becasue 1) we subset around the parcels by several
gridpoints in each direction and 2) don't really care about parcels 
leaving the domain anyway
*/

/* a kernel that applies the lower boundary conditions for
   xvort and yvort*/
__global__ void applyVortBCs(float *xvort, float *yvort, int MX, int MY, int MZ, int tStart, int tEnd, int totTime) { 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if ((i < MX-1) && (j < MY-1) && (k == 0)) { 
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            xvort[arrayIndex(i, j, 0, tidx, MX, MY, MZ)] = xvort[arrayIndex(i, j, 1, tidx, MX, MY, MZ)];
            yvort[arrayIndex(i, j, 0, tidx, MX, MY, MZ)] = yvort[arrayIndex(i, j, 1, tidx, MX, MY, MZ)];
        }
    }
}

/* For the vorticity calculations, you have to do a 4 point average
of the neighbors to get the point onto the scalar grid. Calcvort does
the initial pass, and then this gets called at the end to make sure that
the averaging happens */
__global__ void doVortAvg(float *xvort, float *yvort, float *zvort, int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            xvort[arrayIndex(i, j, k, tidx, MX, MY, MZ)] = 0.25*(xvort[arrayIndex(i, j, k, tidx, MX, MY, MZ)] + \
                xvort[arrayIndex(i, j+1, k, tidx, MX, MY, MZ)] + xvort[arrayIndex(i, j, k+1, tidx, MX, MY, MZ)] + \
                xvort[arrayIndex(i, j+1, k+1, tidx, MX, MY, MZ)]);

            yvort[arrayIndex(i, j, k, tidx, MX, MY, MZ)] = 0.25*(yvort[arrayIndex(i, j, k, tidx, MX, MY, MZ)] + \
                yvort[arrayIndex(i+1, j, k, tidx, MX, MY, MZ)] + yvort[arrayIndex(i, j, k+1, tidx, MX, MY, MZ)] + \
                yvort[arrayIndex(i+1, j, k+1, tidx, MX, MY, MZ)]);

            zvort[arrayIndex(i, j, k, tidx, MX, MY, MZ)] = 0.25*(zvort[arrayIndex(i, j, k, tidx, MX, MY, MZ)] + \
                zvort[arrayIndex(i+1, j, k, tidx, MX, MY, MZ)] + zvort[arrayIndex(i, j+1, k, tidx, MX, MY, MZ)] + \
                zvort[arrayIndex(i+1, j+1, k, tidx, MX, MY, MZ)]);
        }
    }
}

__global__ void doVortTendAvg(float *xvorttilt, float *yvorttilt, float *zvorttilt, int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            xvorttilt[arrayIndex(i, j, k, tidx, MX, MY, MZ)] = 0.5*(xvorttilt[arrayIndex(i, j, k, tidx, MX, MY, MZ)] + \
                xvorttilt[arrayIndex(i+1, j, k, tidx, MX, MY, MZ)]);

            yvorttilt[arrayIndex(i, j, k, tidx, MX, MY, MZ)] = 0.5*(yvorttilt[arrayIndex(i, j, k, tidx, MX, MY, MZ)] + \
                yvorttilt[arrayIndex(i, j+1, k, tidx, MX, MY, MZ)]);

            zvorttilt[arrayIndex(i, j, k, tidx, MX, MY, MZ)] = 0.5*(zvorttilt[arrayIndex(i, j, k, tidx, MX, MY, MZ)] + \
                zvorttilt[arrayIndex(i, j, k+1, tidx, MX, MY, MZ)]);
        }
    }
}


/* Kernel for computing the components of vorticity
    and vorticity forcing terms. We do this using our domain subset containing the parcels
    instead of doing it locally for each parcel, as it would scale poorly for large 
    numbers of parcels. */
__global__ void calcvort(datagrid grid, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                        float *xvort, float *yvort, float *zvort, \
                        int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {

    // get our 3D index based on our blocks/threads
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];
    //printf("%i, %i, %i\n", i, j, k);

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;
            calc_xvort(grid, w_time_chunk, v_time_chunk, xvort, idx_4D, MX, MY, MZ);
            calc_yvort(grid, u_time_chunk, w_time_chunk, yvort, idx_4D, MX, MY, MZ);
            // calculate the Z component of vorticity
            //printf("%i, %i, %i, %i, %i, %i, %i\n", i, j, k, tidx, MX, MY, MZ);
            calc_zvort(grid, u_time_chunk, v_time_chunk, zvort, idx_4D, MX, MY, MZ);

        }
    }
}

/* Calculate the vorticity tendency terms. Needs to be called after the calcvort function, the 4 point average, 
and the boundary conditions have been applied to xvort and yvort. Then this should all be fine? */
__global__ void calcvorttend(datagrid grid, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
							float *xvort, float *yvort, float *zvort, \
							float *xvortstretch, float *yvortstretch, float *zvortstretch, \
							float *xvorttilt, float *yvorttilt, float *zvorttilt, \
                            float *xvortbaro, float *yvortbaro, float *thrhopert, \
							int MX, int MY, int MZ, int tStart, int tEnd, int totTime) {
    // get our 3D index based on our blocks/threads
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;
    int idx_4D[4];
    //printf("%i, %i, %i\n", i, j, k);

    idx_4D[0] = i; idx_4D[1] = j; idx_4D[2] = k;
    
    if ((i < MX-1) && (j < MY-1) && (k < MZ-1)) {
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;

			calc_xvortstretch(grid, xvortstretch, xvort, v_time_chunk, w_time_chunk, idx_4D, MX, MY, MZ );
			calc_yvortstretch(grid, yvortstretch, yvort, u_time_chunk, w_time_chunk, idx_4D, MX, MY, MZ );
			calc_zvortstretch(grid, zvortstretch, zvort, u_time_chunk, v_time_chunk, idx_4D, MX, MY, MZ );
        }
    }

    // these are centered differences and have different boundary requirements
    if ((i < MX-2) && (j < MY-2) && (k < MZ-2) && \
    	(i >= 1) && (j >= 1) && (k >=1)) { 
        if ((i+1 > MX) || (j+1 > MY) || (k+1 > MZ)) printf("i+1 or j+1 out of bounds\n");
        // loop over the number of time steps we have in memory
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            idx_4D[3] = tidx;

			calc_zvorttilt(grid, zvorttilt, xvort, yvort, w_time_chunk, idx_4D, MX, MY, MZ);
			calc_xvorttilt(grid, xvorttilt, yvort, zvort, u_time_chunk, idx_4D, MX, MY, MZ);
			calc_yvorttilt(grid, yvorttilt, xvort, zvort, v_time_chunk, idx_4D, MX, MY, MZ);
            calc_xvortbaro(grid, xvortbaro, thrhopert, idx_4D, MX, MY, MZ);
            calc_yvortbaro(grid, yvortbaro, thrhopert, idx_4D, MX, MY, MZ);
        }
    }
}

__global__ void test(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                    float *p_time_chunk, float *th_time_chunk, \
                    float *xvort_time_chunk, float *yvort_time_chunk, float *zvort_time_chunk, \
                    float *xvorttilt_chunk, float *yvorttilt_chunk, float *zvorttilt_chunk, \
                    float *xvortstretch_chunk, float *yvortstretch_chunk, float *zvortstretch_chunk, \
                    float *xvortbaro_chunk, float *yvortbaro_chunk, \
                    int MX, int MY, int MZ, int tStart, int tEnd, int totTime, int direct) {

	int parcel_id = blockIdx.x;
    // safety check to make sure our thread index doesn't
    // go out of our array bounds
    if (parcel_id < parcels.nParcels) {
        bool is_ugrd = false;
        bool is_vgrd = false;
        bool is_wgrd = false;

        float pcl_u, pcl_v, pcl_w;
        float pcl_ppert, pcl_thrhoprime;
        float pcl_xvort, pcl_yvort, pcl_zvort;
        float pcl_xvorttilt, pcl_yvorttilt, pcl_zvorttilt;
        float pcl_xvortstretch, pcl_yvortstretch, pcl_zvortstretch;
        float pcl_xvortbaro, pcl_yvortbaro;
        float point[3];

        // loop over the number of time steps we are
        // integrating over
        for (int tidx = tStart; tidx < tEnd; ++tidx) {
            // GPU sanity test of data integrity
            point[0] = parcels.xpos[P2(tidx, parcel_id, totTime)];
            point[1] = parcels.ypos[P2(tidx, parcel_id, totTime)];
            point[2] = parcels.zpos[P2(tidx, parcel_id, totTime)];

            is_ugrd = true;
            is_vgrd = false;
            is_wgrd = false;
            pcl_u = interp3D(grid, u_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = true;
            is_wgrd = false;
            pcl_v = interp3D(grid, v_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = true;
            pcl_w = interp3D(grid, w_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            // get the vorticity components
            is_ugrd = false;
            is_vgrd = false;
            is_wgrd = false;
            pcl_xvort = interp3D(grid, xvort_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_yvort = interp3D(grid, yvort_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_zvort = interp3D(grid, zvort_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_xvorttilt = interp3D(grid, xvorttilt_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_yvorttilt = interp3D(grid, yvorttilt_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_zvorttilt = interp3D(grid, zvorttilt_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_xvortstretch = interp3D(grid, xvortstretch_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_yvortstretch = interp3D(grid, yvortstretch_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_zvortstretch = interp3D(grid, zvortstretch_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_xvortbaro = interp3D(grid, xvortbaro_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_yvortbaro = interp3D(grid, yvortbaro_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);

            pcl_ppert = interp3D(grid, p_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);
            pcl_thrhoprime = interp3D(grid, th_time_chunk, point, is_ugrd, is_vgrd, is_wgrd, tidx, MX, MY, MZ);



            // integrate X position forward by the U wind
            point[0] += pcl_u * (1.0f/6.0f) * direct;
            // integrate Y position forward by the V wind
            point[1] += pcl_v * (1.0f/6.0f) * direct;
            // integrate Z position forward by the W wind
            point[2] += pcl_w * (1.0f/6.0f) * direct;
            if ((pcl_u == -999.0) || (pcl_v == -999.0) || (pcl_w == -999.0)) {
                printf("Warning: missing values detected at x: %f y:%f z:%f with ground bounds X0: %f Y0: %f Z0: %f X1: %f Y1: %f Z1: %f\n", \
                    point[0], point[1], point[2], grid.xh[0], grid.yh[0], grid.zh[0], grid.xh[grid.NX-1], grid.yh[grid.NY-1], grid.zh[grid.NZ-1]);
                return;
            }

            if (point[2] < grid.zf[0]) point[2] = grid.zf[0];


            parcels.xpos[P2(tidx+1, parcel_id, totTime)] = point[0]; 
            parcels.ypos[P2(tidx+1, parcel_id, totTime)] = point[1];
            parcels.zpos[P2(tidx+1, parcel_id, totTime)] = point[2];
            parcels.pclu[P2(tidx, parcel_id, totTime)] = pcl_u;
            parcels.pclv[P2(tidx, parcel_id, totTime)] = pcl_v;
            parcels.pclw[P2(tidx, parcel_id, totTime)] = pcl_w;
            parcels.pclppert[P2(tidx, parcel_id, totTime)] = pcl_ppert;
            parcels.pclthrhoprime[P2(tidx, parcel_id, totTime)] = pcl_thrhoprime;

            parcels.pclxvort[P2(tidx, parcel_id, totTime)] = pcl_xvort;
            parcels.pclyvort[P2(tidx, parcel_id, totTime)] = pcl_yvort;
            parcels.pclzvort[P2(tidx, parcel_id, totTime)] = pcl_zvort;
            parcels.pclxvorttilt[P2(tidx, parcel_id, totTime)] = pcl_xvorttilt;
            parcels.pclyvorttilt[P2(tidx, parcel_id, totTime)] = pcl_yvorttilt;
            parcels.pclzvorttilt[P2(tidx, parcel_id, totTime)] = pcl_zvorttilt;
            parcels.pclxvortstretch[P2(tidx, parcel_id, totTime)] = pcl_xvortstretch;
            parcels.pclyvortstretch[P2(tidx, parcel_id, totTime)] = pcl_yvortstretch;
            parcels.pclzvortstretch[P2(tidx, parcel_id, totTime)] = pcl_zvortstretch;
            parcels.pclxvortbaro[P2(tidx, parcel_id, totTime)] = pcl_xvortbaro;
            parcels.pclyvortbaro[P2(tidx, parcel_id, totTime)] = pcl_yvortbaro;
        }
    }
}

/*This function handles allocating memory on the GPU, transferring the CPU
arrays to GPU global memory, calling the integrate GPU kernel, and then
updating the position vectors with the new stuff*/
void cudaIntegrateParcels(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, \
                         float *p_time_chunk, float *th_time_chunk, float *rho_time_chunk, \
                         int MX, int MY, int MZ, int nT, int totTime, int direct) {
    // pointers to device memory
    float *device_u_time_chunk, *device_v_time_chunk, *device_w_time_chunk;
    float *device_p_time_chunk, *device_th_time_chunk, *device_rho_time_chunk;
    float *device_xvort_time_chunk, *device_yvort_time_chunk, *device_zvort_time_chunk;
    float *device_xvortstretch_chunk, *device_yvortstretch_chunk, *device_zvortstretch_chunk;
    float *device_xvorttilt_chunk, *device_yvorttilt_chunk, *device_zvorttilt_chunk;
    float *device_xvortbaro_chunk, *device_yvortbaro_chunk, *device_rhof_time_chunk;
    float *device_t11_time_chunk, *device_t12_time_chunk, *device_t13_time_chunk;
    float *device_t22_time_chunk, *device_t23_time_chunk, *device_t33_time_chunk;

    parcel_pos device_parcels;
    datagrid device_grid;

    int tStart, tEnd;
    tStart = 0;
    tEnd = nT;

    // copy over our integer and long
    // constants to our device struct
    device_grid.X0 = grid.X0; device_grid.X1 = grid.X1;
    device_grid.Y0 = grid.Y0; device_grid.Y1 = grid.Y1;
    device_grid.Z0 = grid.Z0; device_grid.Z1 = grid.Z1;
    device_grid.NX = grid.NX; device_grid.NY = grid.NY;
    device_grid.NZ = grid.NZ; device_grid.nz = grid.nz;
    device_parcels.nParcels = parcels.nParcels;

    // allocate device memory for our grid arrays
    gpuErrchk( cudaMalloc(&(device_grid.xh), device_grid.NX*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.yh), device_grid.NY*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.zh), device_grid.NZ*sizeof(float)) );

    gpuErrchk( cudaMalloc(&(device_grid.th0), device_grid.nz*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.qv0), device_grid.nz*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.rho0), device_grid.nz*sizeof(float)) );

    gpuErrchk( cudaMalloc(&(device_grid.xf), device_grid.NX*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.yf), device_grid.NY*sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_grid.zf), device_grid.NZ*sizeof(float)) );
    // allocate the device memory for U/V/W
    gpuErrchk( cudaMalloc(&device_u_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_v_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_w_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_p_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_th_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_rho_time_chunk, MX*MY*MZ*nT*sizeof(float)) );

    //vorticity device arrays we have to calculate
    gpuErrchk( cudaMalloc(&device_xvort_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_yvort_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_zvort_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    // vorticity tendency arrays
    gpuErrchk( cudaMalloc(&device_xvortstretch_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_yvortstretch_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_zvortstretch_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_xvorttilt_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_yvorttilt_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_zvorttilt_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_xvortbaro_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_yvortbaro_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_rhof_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    // turbulent stress arrays
    gpuErrchk( cudaMalloc(&device_t11_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_t12_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_t13_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_t22_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_t23_time_chunk, MX*MY*MZ*nT*sizeof(float)) );
    gpuErrchk( cudaMalloc(&device_t33_time_chunk, MX*MY*MZ*nT*sizeof(float)) );

    // allocate device memory for our parcel positions
    gpuErrchk( cudaMalloc(&(device_parcels.xpos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.ypos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.zpos), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclu), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclv), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclw), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclppert), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclthrhoprime), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclxvort), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclyvort), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclzvort), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclxvorttilt), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclyvorttilt), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclzvorttilt), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclxvortstretch), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclyvortstretch), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclzvortstretch), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclxvortbaro), parcels.nParcels * totTime * sizeof(float)) );
    gpuErrchk( cudaMalloc(&(device_parcels.pclyvortbaro), parcels.nParcels * totTime * sizeof(float)) );

    // copy the arrays to device memory
    gpuErrchk( cudaMemcpy(device_u_time_chunk, u_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_v_time_chunk, v_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_w_time_chunk, w_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_p_time_chunk, p_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_th_time_chunk, th_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_rho_time_chunk, rho_time_chunk, MX*MY*MZ*nT*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_parcels.xpos, parcels.xpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.ypos, parcels.ypos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_parcels.zpos, parcels.zpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_grid.xh, grid.xh, device_grid.NX*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.yh, grid.yh, device_grid.NY*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.zh, grid.zh, device_grid.NZ*sizeof(float), cudaMemcpyHostToDevice) );
    
    gpuErrchk( cudaMemcpy(device_grid.th0, grid.th0, device_grid.nz*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.qv0, grid.qv0, device_grid.nz*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.rho0, grid.rho0, device_grid.nz*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(device_grid.xf, grid.xf, device_grid.NX*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.yf, grid.yf, device_grid.NY*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(device_grid.zf, grid.zf, device_grid.NZ*sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaDeviceSynchronize() );
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((MX/threadsPerBlock.x)+1, (MY/threadsPerBlock.y)+1, (MZ/threadsPerBlock.z)+1); 

    cout << "Calculating rhof" << endl;

    doCalcrf<<<numBlocks, threadsPerBlock>>>(device_grid, device_rho_time_chunk, device_rhof_time_chunk, MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );

    cout << "Calculating Deformation for Turbulence" << endl;
    doCalcdef<<<numBlocks, threadsPerBlock>>>(device_grid, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, device_rho_time_chunk, device_rhof_time_chunk, \
                        device_t11_time_chunk, device_t12_time_chunk, device_t13_time_chunk, device_t22_time_chunk, device_t23_time_chunk, device_t33_time_chunk, \
                        MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );

    cout << "Calculating vorticity" << endl;
    calcvort<<<numBlocks, threadsPerBlock>>>(device_grid, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, \
                                            device_xvort_time_chunk, device_yvort_time_chunk, device_zvort_time_chunk, \
                                            MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );

    applyVortBCs<<<numBlocks, threadsPerBlock>>>(device_xvort_time_chunk, device_yvort_time_chunk, MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );
    
    doVortAvg<<<numBlocks, threadsPerBlock>>>(device_xvort_time_chunk, device_yvort_time_chunk, device_zvort_time_chunk, MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );
    cout << "doing tendencies" << endl;
    calcvorttend<<<numBlocks, threadsPerBlock>>>(device_grid, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, \
    				device_xvort_time_chunk, device_yvort_time_chunk, device_zvort_time_chunk, \
    				device_xvortstretch_chunk, device_yvortstretch_chunk, device_zvortstretch_chunk, \
    				device_xvorttilt_chunk, device_yvorttilt_chunk, device_zvorttilt_chunk, \
                    device_xvortbaro_chunk, device_yvortbaro_chunk, device_th_time_chunk, \
    				MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );

    doVortTendAvg<<<numBlocks, threadsPerBlock>>>(device_xvorttilt_chunk, device_yvorttilt_chunk, device_zvorttilt_chunk, MX, MY, MZ, tStart, tEnd, totTime);
    gpuErrchk( cudaDeviceSynchronize() );
    cout << "ending tendencies" << endl;
    
    cout << "End vorticity calc" << endl;
    test<<<parcels.nParcels,1>>>(device_grid, device_parcels, device_u_time_chunk, device_v_time_chunk, device_w_time_chunk, \
                                device_p_time_chunk, device_th_time_chunk, \
                                device_xvort_time_chunk, device_yvort_time_chunk, device_zvort_time_chunk, \
                                device_xvorttilt_chunk, device_yvorttilt_chunk, device_zvorttilt_chunk, \
                                device_xvortstretch_chunk, device_yvortstretch_chunk, device_zvortstretch_chunk, \
                                device_xvortbaro_chunk, device_yvortbaro_chunk, \
                                MX, MY, MZ, tStart, tEnd, totTime, direct);
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(parcels.xpos, device_parcels.xpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.ypos, device_parcels.ypos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.zpos, device_parcels.zpos, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclu, device_parcels.pclu, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclv, device_parcels.pclv, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclw, device_parcels.pclw, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclppert, device_parcels.pclppert, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclthrhoprime, device_parcels.pclthrhoprime, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclxvort, device_parcels.pclxvort, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclyvort, device_parcels.pclyvort, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclzvort, device_parcels.pclzvort, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclxvorttilt, device_parcels.pclxvorttilt, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclyvorttilt, device_parcels.pclyvorttilt, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclzvorttilt, device_parcels.pclzvorttilt, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclxvortstretch, device_parcels.pclxvortstretch, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclyvortstretch, device_parcels.pclyvortstretch, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclzvortstretch, device_parcels.pclzvortstretch, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclxvortbaro, device_parcels.pclxvortbaro, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(parcels.pclyvortbaro, device_parcels.pclyvortbaro, parcels.nParcels * totTime * sizeof(float), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaDeviceSynchronize() );

    cudaFree(device_grid.xh);
    cudaFree(device_grid.yh);
    cudaFree(device_grid.zh);
    cudaFree(device_grid.xf);
    cudaFree(device_grid.yf);
    cudaFree(device_grid.zf);
    cudaFree(device_grid.th0);
    cudaFree(device_grid.qv0);
    cudaFree(device_grid.rho0);
    cudaFree(device_parcels.xpos);
    cudaFree(device_parcels.ypos);
    cudaFree(device_parcels.zpos);
    cudaFree(device_parcels.pclu);
    cudaFree(device_parcels.pclv);
    cudaFree(device_parcels.pclw);
    cudaFree(device_parcels.pclppert);
    cudaFree(device_parcels.pclthrhoprime);
    cudaFree(device_parcels.pclxvort);
    cudaFree(device_parcels.pclyvort);
    cudaFree(device_parcels.pclzvort);
    cudaFree(device_parcels.pclxvorttilt);
    cudaFree(device_parcels.pclyvorttilt);
    cudaFree(device_parcels.pclzvorttilt);
    cudaFree(device_parcels.pclxvortstretch);
    cudaFree(device_parcels.pclyvortstretch);
    cudaFree(device_parcels.pclzvortstretch);
    cudaFree(device_parcels.pclxvortbaro);
    cudaFree(device_parcels.pclyvortbaro);
    cudaFree(device_u_time_chunk);
    cudaFree(device_v_time_chunk);
    cudaFree(device_w_time_chunk);
    cudaFree(device_p_time_chunk);
    cudaFree(device_th_time_chunk);
    cudaFree(device_rho_time_chunk);
    cudaFree(device_xvort_time_chunk);
    cudaFree(device_yvort_time_chunk);
    cudaFree(device_zvort_time_chunk);
    cudaFree(device_xvorttilt_chunk);
    cudaFree(device_yvorttilt_chunk);
	cudaFree(device_zvorttilt_chunk);
    cudaFree(device_xvortstretch_chunk);
    cudaFree(device_yvortstretch_chunk);
	cudaFree(device_zvortstretch_chunk);
	cudaFree(device_xvortbaro_chunk);
	cudaFree(device_yvortbaro_chunk);
	cudaFree(device_rhof_time_chunk);
	cudaFree(device_t11_time_chunk);
	cudaFree(device_t12_time_chunk);
	cudaFree(device_t13_time_chunk);
	cudaFree(device_t22_time_chunk);
	cudaFree(device_t23_time_chunk);
	cudaFree(device_t33_time_chunk);



    gpuErrchk( cudaDeviceSynchronize() );
    cout << "FINISHED CUDA" << endl;
}

#endif
