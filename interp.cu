#include <iostream>
#include "math.h"
using namespace std;

// error checking helper
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      cout << cudaGetErrorString(code) << endl;
      if (abort) exit(code);
   }
}


// We are not working with full integration time data - some sort of subset of 
// data from CM1 is expected. We expect there to be U, V, and W on native (uninterpolated)
// grids, information about the X, Y, and Z grid points, and a set of scalar fields.
// It is up to the user to specify grid spacing at run-time. It is assumed that there is
// no grid stretching at this time (i.e. grid scale factor = 1.0) 

__device__ float DX = 30.; 
__device__ float DY = 30.; 
__device__ float DZ = 30.;
__device__ float scale_x = 1.0;
__device__ float scale_y = 1.0;
__device__ float scale_z = 1.0;


// Compute the nearest grid point index along each dimension
// for a single parcel.
__device__ int* _nearest_grid_idx(float pt_x, float pt_y, float pt_z, \
								float *x_grd, float *y_grd, float *z_grd, \
								int nX, int nY, int nZ) {

	int *idx_3D = new int[3];
	int near_i = -1;
	int near_j = -1;
	int near_k = -1;
	// get thread index
	//int pidx = blockIdx.x*blockDim.x + threadIdx.x;


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
	if ((near_i == -1) | (near_j == -1) | (near_k) == -1) {
		near_i = -1; near_j = -1; near_k = -1;
	}

	idx_3D[0] = near_i; idx_3D[1] = near_j; idx_3D[2] = near_k;
	return idx_3D;
}

// calculate the 8 weights for a single parcel
__device__ float* _calc_weights(float *x_grd, float *y_grd, float *z_grd, float x_pt, float y_pt, \
								float z_pt, bool ugrd, bool vgrd, bool wgrd, int nX, int nY, int nZ) {
	int i, j, k;
	float rx, ry, rz;
	float w1, w2, w3, w4;
	float w5, w6, w7, w8;
	float *weights = new float[8];
	
	// initialize the weights to -1
	// to be returned in the event that
	// the requested grid point is out
	// of the bounds of the domain
	for (int i = 0; i < 8; i++) {
		weights[i] = -1;
	}

	// calculate the 3D index of the nearest grid point
	int* idx_3D = _nearest_grid_idx(x_pt, y_pt, z_pt, x_grd, y_grd, z_grd, nX, nY, nZ);

	// check to see if the requested point is within the grid domain.
	// If any grid index is out of bounds, immediately return all weights
	// as -1.
	for (int i = 0; i < 3; i++) {
		if (idx_3D[i] == -1) {
			return weights;
		}
	}

	// store the indices of the nearest grid point
	// in i, j, k
	i = idx_3D[0]; j = idx_3D[1]; k = idx_3D[2];

	// the U, V, and W grids are staggered so this
	// takes care of that crap
	if (ugrd) {
		rx = (x_pt - x_grd[i] + 0.5 * DX / scale_x) * scale_x / DX;
		ry = (y_pt - y_grd[j]) * scale_y / DY;
		rz = (z_pt - z_grd[k]) * scale_z / DZ;
	}

	if (vgrd) {
		rx = (x_pt - x_grd[i]) * scale_x / DX;
		ry = (y_pt - y_grd[j] + 0.5 * DX / scale_y) * scale_y / DY;
		rz = (z_pt - z_grd[k]) * scale_z / DZ;

	}

	if (wgrd) {
		rx = (x_pt - x_grd[i]) * scale_x / DX;
		ry = (y_pt - y_grd[j]) * scale_y / DY;
		rz = (z_pt - z_grd[k] - 0.5 * DZ  / scale_z) * scale_z / DZ;

	}

	// data on scalar grid
	else {
		rx = (x_pt - x_grd[i]) * scale_x / DX;
		ry = (y_pt - y_grd[j]) * scale_y / DY;
		rz = (z_pt - z_grd[k]) * scale_z / DZ;
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

	return weights;

}
