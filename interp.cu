#include <iostream>
#include "math.h"
using namespace std;

#ifndef INTERP
#define INTERP
// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
#define P4(x,y,z,t,mx,my,mz) (((t)*(mx)*(my)*(mz))+((z)*(mx)*(my))+((y)*(mx))+(x))
__host__ __device__ int arrayIndex(int x, int y, int z, int t,  int mx, int my, int mz) {
	return P4(x, y, z, t, mx, my, mz);
}


// We are not working with full integration time data - some sort of subset of 
// data from CM1 is expected. We expect there to be U, V, and W on native (uninterpolated)
// grids, information about the X, Y, and Z grid points, and a set of scalar fields.
// It is up to the user to specify grid spacing at run-time. It is assumed that there is
// no grid stretching at this time (i.e. grid scale factor = 1.0) 
static const float DX = 30.; 
static const float DY = 30.; 
static const float DZ = 30.;
static const float scale_x = 1.0;
static const float scale_y = 1.0;
static const float scale_z = 1.0;


// find the nearest grid index i, j, and k for a point contained inside of a cube.
// i, j, and k are set to -1 if the point requested is out of the domain bounds
// of the cube provided.
__device__ __host__ void _nearest_grid_idx(float *point, float *x_grd, float *y_grd, float *z_grd, \
                                 int *idx_4D, int nX, int nY, int nZ) {

	int near_i = -1;
	int near_j = -1;
	int near_k = -1;

    float pt_x = point[0];
    float pt_y = point[1];
    float pt_z = point[2];


	// loop over the X grid
	for ( int i = 0; i < nX; i++ ) {
		// find the nearest grid point index at X
		if ( fabs( pt_x - x_grd[i] ) <= ( 0.5 * DX / scale_x ) ) { near_i = i; }
	}

	// loop over the Y grid
	for ( int j = 0; j < nY; j++ ) {
		// find the nearest grid point index in the Y
		if ( fabs( pt_y - y_grd[j] ) <= ( 0.5 * DY / scale_y ) ) { near_j = j; }
	}

	// loop over the Z grid
	for (int k = 0; k < nZ; k++ ) {
		// find the nearest grid point index in the Z
		if ( fabs( pt_z - z_grd[k] ) <= ( 0.5 * DZ / scale_z ) ) { near_k = k; }
	}

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
__host__ __device__ void _calc_weights(float *x_grd, float *y_grd, float *z_grd, float *weights, \
                                       float *point, int *idx_4D, bool ugrd, bool vgrd, bool wgrd, \
                                       int nX, int nY, int nZ) {
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
		weights[i] = -1;
	}

	// check to see if the requested point is within the grid domain.
	// If any grid index is out of bounds, immediately return all weights
	// as -1.
	for (int i = 0; i < 3; i++) {
		if (idx_4D[i] == -1) {
			return;
		}
	}

	// store the indices of the nearest grid point
	// in i, j, k
	i = idx_4D[0]; j = idx_4D[1]; k = idx_4D[2];

	// the U, V, and W grids are staggered so this
	// takes care of that crap
	if (ugrd) {
		rx = ((x_pt - x_grd[i]) + (0.5 * DX / scale_x)) * (scale_x / DX);
		ry = ((y_pt - y_grd[j])) * (scale_y / DY);
		rz = ((z_pt - z_grd[k])) * (scale_z / DZ);
	}

	if (vgrd) {
		rx = ((x_pt - x_grd[i])) * (scale_x / DX);
		ry = ((y_pt - y_grd[j] + (0.5 * DX / scale_y))) * (scale_y / DY);
		rz = ((z_pt - z_grd[k])) * (scale_z / DZ);

	}

	if (wgrd) {
		rx = ((x_pt - x_grd[i])) * (scale_x / DX);
		ry = ((y_pt - y_grd[j])) * (scale_y / DY);
		rz = ((z_pt - z_grd[k]) - (0.5 * DZ  / scale_z)) * (scale_z / DZ);

	}

	// data on scalar grid
	else {
		rx = (x_pt - x_grd[i]) * (scale_x / DX);
		ry = (y_pt - y_grd[j]) * (scale_y / DY);
		rz = (z_pt - z_grd[k]) * (scale_z / DZ);
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
__host__ __device__ float _tri_interp(float *data_arr, float* weights, int *idx_4D, int NX, int NY, int NZ) {
	float out = -999.0;
	int idx1, idx2, idx3, idx4;
	int idx5, idx6, idx7, idx8;

    int i = idx_4D[0]; int j = idx_4D[1]; int k = idx_4D[2]; int t = idx_4D[3];

	// if the given i,j,k are invalid, return -999.0
	if ((i == -1) | (j == -1) | (k == -1)) {
		return out;
	}

	// if the given weights are invalid, return -999.0
	for (int idx = 0; idx < 8; idx++) {
		if (weights[idx] == -1) {
			return out;
		}
	}


	// from here on our, we assume out point is inside of the domain,
	// and there are weights with values between 0 and 1.

	// get the array indices
	idx1 = arrayIndex(i, j, k, t, NX, NY, NZ);
	idx2 = arrayIndex(i+1, j, k, t, NX, NY, NZ);
	idx3 = arrayIndex(i, j+1, k, t, NX, NY, NZ);
	idx4 = arrayIndex(i, j, k+1, t, NX, NY, NZ);
	idx5 = arrayIndex(i+1, j, k+1, t, NX, NY, NZ);
	idx6 = arrayIndex(i, j+1, k+1, t, NX, NY, NZ);
	idx7 = arrayIndex(i+1, j+1, k, t, NX, NY, NZ);
	idx8 = arrayIndex(i+1, j+1, k+1, t, NX, NY, NZ);

	out = data_arr[idx1] * weights[0] + \
		  data_arr[idx2] * weights[1] + \
		  data_arr[idx3] * weights[2] + \
		  data_arr[idx4] * weights[3] + \
		  data_arr[idx5] * weights[4] + \
		  data_arr[idx6] * weights[5] + \
		  data_arr[idx7] * weights[6] + \
		  data_arr[idx8] * weights[7];
	return out;

}


// wrapper function around all of the necessary components for 3D interpolation. Calls the function that finds
// the nearest grid point, calculates the interpolation weights depending on whether or not the grid is staggered,
// and then calls the trilinear interpolator. Returns -999.0 if the data is not inside the grid or the weights
// are invalid.
__host__ __device__ float interp3D(float *x_grd, float *y_grd, float *z_grd, float *data_grd, float *point, \
                                    bool ugrd, bool vgrd, bool wgrd, int tstep, int nX, int nY, int nZ) {
    int idx_4D[4];
    float weights[8];
    float output_val;

    idx_4D[3] = tstep;

    // get the index of the nearest grid point to the
    // data we are requesting
    _nearest_grid_idx(point, x_grd, y_grd, z_grd, idx_4D, nX, nY, nZ);

    // get the interpolation weights
    _calc_weights(x_grd, y_grd, z_grd, weights, point, idx_4D, ugrd, vgrd, wgrd, nX, nY, nZ); 

    // interpolate the value
    output_val = _tri_interp(data_grd, weights, idx_4D, nX, nY, nZ);

    return output_val;
}

#endif
