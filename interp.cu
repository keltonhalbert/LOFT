#include <iostream>
#include <stdio.h>
#include "math.h"
#include "macros.cpp"
using namespace std;

#ifndef INTERP
#define INTERP

// find the nearest grid index i, j, and k for a point contained inside of a cube.
// i, j, and k are set to -1 if the point requested is out of the domain bounds
// of the cube provided.
__device__ __host__ void _nearest_grid_idx(float *point, datagrid *grid, int *idx_4D) {

	int near_i = -1;
	int near_j = -1;
	int near_k = -1;

    float pt_x = point[0];
    float pt_y = point[1];
    float pt_z = point[2];


	// loop over the X grid
	for ( int i = 0; i < grid->NX; i++ ) {
		// find the nearest grid point index at X
		if ( ( pt_x >= grid->xf[i] ) && ( pt_x <= grid->xf[i+1] ) ) { near_i = i; } 
	}

	// loop over the Y grid
	for ( int j = 0; j < grid->NY; j++ ) {
		// find the nearest grid point index in the Y
		if ( ( pt_y >= grid->yf[j] ) && ( pt_y <= grid->yf[j+1] ) ) { near_j = j; } 
	}

	// loop over the Z grid
    int k = 0;
    while (pt_z >= grid->zf[k+1]) {
        k = k + 1;
    }
    near_k = k;

	// if a nearest index was not found, set all indices to -1 to flag
	// that the point is not in the domain
	if ((near_i == -1) || (near_j == -1) || (near_k == -1)) {
		near_i = -1; near_j = -1; near_k = -1;
	}

	idx_4D[0] = near_i; idx_4D[1] = near_j; idx_4D[2] = near_k;
	return;
}

// calculate the 8 interpolation weights for a trilinear interpolation of a point inside of a cube.
// Returns an array full of -1 if the requested poit is out of the domain bounds
__host__ __device__ void _calc_weights(datagrid *grid, float *weights, float *point, \
                                       int *idx_4D, bool ugrd, bool vgrd, bool wgrd) {
	int i, j, k;
	float rx, ry, rz;
	float w1, w2, w3, w4;
	float w5, w6, w7, w8;

    float x_pt = point[0]; float y_pt = point[1]; float z_pt = point[2];
	
	// initialize the weights to -1
	// to be returned in the event that
	// the requested grid point is out
	// of the bounds of the domain
	for (int i = 0; i < 8; i++) {
		weights[i] = -999;
	}

	// check to see if the requested point is within the grid domain.
	// If any grid index is out of bounds, immediately return all weights
	// as missing.
	for (int i = 0; i < 3; i++) {
		if (idx_4D[i] == -1) {
			return;
		}
	}

	// the U, V, and W grids are staggered so this
	// takes care of that crap, as well as handling
    // the changes between staggered and unstaggered 
    // meshes
	if (ugrd) {
        if (y_pt < grid->yh[idx_4D[1]]) {
            idx_4D[1] = idx_4D[1] - 1;
        }
        if ( (z_pt < grid->zh[idx_4D[2]]) && (idx_4D[2] != 0) ) {
            idx_4D[2] = idx_4D[2] - 1;
        }
        i = idx_4D[0]; j = idx_4D[1]; k = idx_4D[2];

        rx = (x_pt - grid->xf[i]) / (grid->xf[i+1] - grid->xf[i]); 
        ry = (y_pt - grid->yh[j]) / (grid->yh[j+1] - grid->yh[j]); 
        rz = (z_pt - grid->zh[k]) / (grid->zh[k+1] - grid->zh[k]); 
        
	}

    else if (vgrd) {
        if (x_pt < grid->xh[idx_4D[0]]) {
            idx_4D[0] = idx_4D[0] - 1;
        }
        if ( (z_pt < grid->zh[idx_4D[2]]) && (idx_4D[2] != 0) ) {
            idx_4D[2] = idx_4D[2] - 1;
        }
        i = idx_4D[0]; j = idx_4D[1]; k = idx_4D[2];


        rx = (x_pt - grid->xh[i]) / (grid->xh[i+1] - grid->xh[i]); 
        ry = (y_pt - grid->yf[j]) / (grid->yf[j+1] - grid->yf[j]); 
        rz = (z_pt - grid->zh[k]) / (grid->zh[k+1] - grid->zh[k]); 

	}

    else if (wgrd) {
        if (x_pt < grid->xh[idx_4D[0]]) {
            idx_4D[0] = idx_4D[0] - 1;
        }
        if (y_pt < grid->yh[idx_4D[1]]) {
            idx_4D[1] = idx_4D[1] - 1;
        }
        i = idx_4D[0]; j = idx_4D[1]; k = idx_4D[2];
        rx = (x_pt - grid->xh[i]) / (grid->xh[i+1] - grid->xh[i]); 
        ry = (y_pt - grid->yh[j]) / (grid->yh[j+1] - grid->yh[j]); 
        rz = (z_pt - grid->zf[k]) / (grid->zf[k+1] - grid->zf[k]); 

	}

    // data is on scalar grid
    else {
        if (x_pt < grid->xh[idx_4D[0]]) {
            idx_4D[0] = idx_4D[0] - 1;
        }
        if (y_pt < grid->yh[idx_4D[1]]) {
            idx_4D[1] = idx_4D[1] - 1;
        }
        if ( (z_pt < grid->zh[idx_4D[2]]) && (idx_4D[2] != 0) ) {
            idx_4D[2] = idx_4D[2] - 1;
        }
        i = idx_4D[0]; j = idx_4D[1]; k = idx_4D[2];
    
        rx = (x_pt - grid->xh[i]) / (grid->xh[i+1] - grid->xh[i]); 
        ry = (y_pt - grid->yh[j]) / (grid->yh[j+1] - grid->yh[j]); 
        rz = (z_pt - grid->zh[k]) / (grid->zh[k+1] - grid->zh[k]); 
    }


	// calculate the weights
    w1 = (1.0 - rx) * (1.0 - ry) * (1.0 - rz);
    w2 = rx * (1.0 - ry) * (1.0 - rz);
    w3 = (1.0 - rx) * ry * (1.0 - rz);
    w4 = (1.0 - rx) * (1.0 - ry) * rz;
    w5 = rx * (1.0 - ry) * rz;
    w6 = (1.0 - rx) * ry * rz;
    w7 = rx * ry * (1.0 - rz);
    w8 = rx * ry * rz;

    // assign the weights to the
    // array of weights;
    weights[0] = w1;
    weights[1] = w2;
    weights[2] = w3;
    weights[3] = w4;
    weights[4] = w5;
    weights[5] = w6;
    weights[6] = w7;
    weights[7] = w8;

}


// interpolate the value of a point contained within a 3D grid.

// data_arr is a 3D field allocated into a contiguous 1D array block of memory.
// weights is a 1D array of interpolation weights returned by _calc_weights
// idx_3D containing the i, j, and k are the respective indices of the nearest grid point we are
// interpolating to, returned by _nearest_grid_idx 
__host__ __device__ float _tri_interp(float *data_arr, float* weights, bool ugrd, bool vgrd, bool wgrd,\
                                        int *idx_4D, int NX, int NY, int NZ) {
	float out = -999.0;

    int i = idx_4D[0]; int j = idx_4D[1]; int k = idx_4D[2]; int t = idx_4D[3];

	// if the given i,j,k are invalid, return -999.0
	if ((i == -1) | (j == -1) | (k == -1)) {
		return out;
	}

	// if the given weights are invalid, return -999.0
	for (int idx = 0; idx < 8; idx++) {
		if (weights[idx] == -999) {
			return out;
		}
	}


	// from here on out, we assume out point is inside of the domain,
	// and there are weights with values between 0 and 1.

    if (ugrd) {
        //printf("I'm a U staggered interpolation!\n");
        float *ustag = data_arr;
        out = (UA4D(i ,  j, k  , t) * weights[0]) + \
              (UA4D(i+1, j, k  , t) * weights[1]) + \
              (UA4D(i ,j+1, k  , t) * weights[2]) + \
              (UA4D(i ,j  , k+1, t) * weights[3]) + \
              (UA4D(i+1, j, k+1, t) * weights[4]) + \
              (UA4D(i  ,j+1,k+1, t) * weights[5]) + \
              (UA4D(i+1,j+1, k,  t) * weights[6]) + \
              (UA4D(i+1,j+1, k+1,t) * weights[7]);
    }
    else if (vgrd) {
        //printf("I'm a V staggered interpolation!\n");
        float *vstag = data_arr;
        out = (VA4D(i ,  j, k  , t) * weights[0]) + \
              (VA4D(i+1, j, k  , t) * weights[1]) + \
              (VA4D(i ,j+1, k  , t) * weights[2]) + \
              (VA4D(i ,j  , k+1, t) * weights[3]) + \
              (VA4D(i+1, j, k+1, t) * weights[4]) + \
              (VA4D(i  ,j+1,k+1, t) * weights[5]) + \
              (VA4D(i+1,j+1, k,  t) * weights[6]) + \
              (VA4D(i+1,j+1, k+1,t) * weights[7]);
        }
    else if (wgrd) {
        //printf("I'm a W staggered interpolation!\n");
        float *wstag = data_arr;
        out = (WA4D(i ,  j, k  , t) * weights[0]) + \
              (WA4D(i+1, j, k  , t) * weights[1]) + \
              (WA4D(i ,j+1, k  , t) * weights[2]) + \
              (WA4D(i ,j  , k+1, t) * weights[3]) + \
              (WA4D(i+1, j, k+1, t) * weights[4]) + \
              (WA4D(i  ,j+1,k+1, t) * weights[5]) + \
              (WA4D(i+1,j+1, k,  t) * weights[6]) + \
              (WA4D(i+1,j+1, k+1,t) * weights[7]);
    }
    else {
        float *buf0 = data_arr;
        //printf("I'm a scalar interpolation!\n");
        out = (BUF4D(i ,  j, k  , t) * weights[0]) + \
              (BUF4D(i+1, j, k  , t) * weights[1]) + \
              (BUF4D(i ,j+1, k  , t) * weights[2]) + \
              (BUF4D(i ,j  , k+1, t) * weights[3]) + \
              (BUF4D(i+1, j, k+1, t) * weights[4]) + \
              (BUF4D(i  ,j+1,k+1, t) * weights[5]) + \
              (BUF4D(i+1,j+1, k,  t) * weights[6]) + \
              (BUF4D(i+1,j+1, k+1,t) * weights[7]);
        }
	return out;

}


// wrapper function around all of the necessary components for 3D interpolation. Calls the function that finds
// the nearest grid point, calculates the interpolation weights depending on whether or not the grid is staggered,
// and then calls the trilinear interpolator. Returns -999.0 if the data is not inside the grid or the weights
// are invalid.
__host__ __device__ float interp3D(datagrid *grid, float *data_grd, float *point, \
                                    bool ugrd, bool vgrd, bool wgrd, int tstep) {
    int idx_4D[4];
    float weights[8];
    float output_val;

    idx_4D[3] = tstep;

    // get the index of the nearest grid point to the
    // data we are requesting
    _nearest_grid_idx(point, grid, idx_4D);

    // get the interpolation weights
    _calc_weights(grid, weights, point, idx_4D, ugrd, vgrd, wgrd); 

    // interpolate the value
    output_val = _tri_interp(data_grd, weights, ugrd, vgrd, wgrd, idx_4D, grid->NX, grid->NY, grid->NZ);

    if (output_val == -999.0) {
        printf("val = %f x = %f y = %f z = %f i = %d j = %d k = %d\n", output_val, point[0], point[1], point[2], idx_4D[0], idx_4D[1], idx_4D[2]);
    }

    return output_val;
}

#endif
