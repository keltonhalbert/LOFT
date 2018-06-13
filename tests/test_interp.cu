#include "../interp.cu"
#include <stdio.h>
using namespace std;


// This function tests the _nearest_grid_idx method in order 
// to determine that things are working as intended.
// It creates a small 30m isotropic grid and calls the GPU code
// to find the nearest grid point index, and then returns the
// grid point information for that index. This is a GPU method.
void _testNearestIndex(float *point, float *nearest_point) {
	printf("INPUT PARAMATERS\n");
	printf("%0.6f %0.6f %0.6f\n", point[0], point[1], point[2]);
	printf("%0.6f %0.6f %0.6f\n\n", nearest_point[0], nearest_point[1], nearest_point[2]);

	int idx[3];
	float *x_grd, *y_grd, *z_grd;
	float near_x, near_y, near_z;
	int pi, pj, pk;
	int nX = 10;
	int nY = 10;
	int nZ = 10;

	// initialize our fake 3D grid with a 30m grid spacing
	// out 10 grid points in each direction
	x_grd = new float[nX];
	for (int i = 0; i < nX; i++) {
		x_grd[i] = i*30.;
	}

	y_grd = new float[nY];
	for(int j = 0; j < nY; j++) {
		y_grd[j] = j*30.;
	}

	z_grd = new float[nZ];
	for (int k = 0; k < nZ; k++) {
		z_grd[k] = k*30.;
	}


	// get the nearest point index
	_nearest_grid_idx(point, x_grd, y_grd, z_grd, idx, nX, nY, nZ);
	// use the index to get the nearest grid point values
	pi= idx[0];
	pj = idx[1];
	pk = idx[2];

	if ((pi == -1) | (pj == -1) | (pk == -1)) {
		near_x = -999.99;
		near_y = -999.99;
		near_z = -999.99;
	}

	else {
		near_x = x_grd[pi];
		near_y = y_grd[pj];
		near_z = z_grd[pk];
	}

	nearest_point[0] = near_x;
	nearest_point[1] = near_y;
	nearest_point[2] = near_z;
	// print the output
	//cout << "X input: " << pt_x << " | Nearest X Point: " << near_x << endl; 

}

// Test whether or not the _calc_weights function works as intended.
// Creates a small 30m isotropic grid to test the weigting function.
// _calc_weights calls _nearest_grid_idx. This is a GPU method. 
void _testCalcWeights(float *point, float *weights_out) {

	float *x_grd, *y_grd, *z_grd;
	float weights[8];
    int idx_3D[3];
	int nX = 10;
	int nY = 10;
	int nZ = 10;

	// initialize our fake 3D grid with a 30m grid spacing
	// out 10 grid points in each direction
	x_grd = new float[nX];
	for (int i = 0; i < nX; i++) {
		x_grd[i] = i*30.;
	}

	y_grd = new float[nY];
	for(int j = 0; j < nY; j++) {
		y_grd[j] = j*30.;
	}

	z_grd = new float[nZ];
	for (int k = 0; k < nZ; k++) {
		z_grd[k] = k*30.;
	}


	// right now we just care about testing it on the unstaggered grid
	// since we have a regular grid. Probably should test grid staggering
	// at some point, but nah.

	// get the nearest point index
	_nearest_grid_idx(point, x_grd, y_grd, z_grd, idx_3D, nX, nY, nZ);
    _calc_weights(x_grd, y_grd, z_grd, weights, point, idx_3D,  false, false, false, nX, nY, nZ);
	for (int i = 0; i < 8; i++) {
		weights_out[i] = weights[i];
	}

}

void _testTriInterp(float *point, float *interp_val) {

    float *x_grd, *y_grd, *z_grd;
    int nX = 10;
    int nY = 10;
    int nZ = 10;
    int N = nX*nY*nZ;

    float *data_arr = new float[N];

	// initialize our fake 3D grid with a 30m grid spacing
	// out 10 grid points in each direction
	x_grd = new float[nX];
	for (int i = 0; i < nX; i++) {
		x_grd[i] = i*30.;
	}

	y_grd = new float[nY];
	for(int j = 0; j < nY; j++) {
		y_grd[j] = j*30.;
	}

	z_grd = new float[nZ];
	for (int k = 0; k < nZ; k++) {
		z_grd[k] = k*30.;
	}

    //initialize our fake data grid with ones in all dimensions
    for ( int i = 0; i < nX; ++i) {
        for (int j = 0; j < nY; ++j) {
            for (int k = 0; k < nZ; ++k) {
            data_arr[arrayIndex(i, j, k, nX, nY)] = k;
            }
        }
    } 

    float result = interp3D(x_grd, y_grd, z_grd, data_arr, point, false, false, false, nX, nY, nZ);

    interp_val[0] = result; 

    return;

}
//__host__ __device__ float _tri_interp(float* data_arr, int NX, int NY, float* weights, int i, int j, int k) {


// The CPU method for testing the GPU method that returns the nearest grid index
void testNearestIndex() {
	float *nearest_point = new float[3];
	float *point = new float[3];


	cout << endl << "TESTING THE NEAREST GRIDPOINT FINDER USING 30 METER ISOTROPIC" << endl;
	cout << "CASE WHERE DATA IS EXACTLY ON GRID" << endl;
	point[0] = 30.; point[1] = 30.; point[2] = 30.;
	nearest_point[0] = -1.; nearest_point[1] = -1.; nearest_point[2] = -1.;
	_testNearestIndex(point, nearest_point);
	cout << "OUTPUT" << endl;
	cout << "X input: " << point[0] << " | Nearest X Point: " << nearest_point[0] << endl;
	cout << "Y input: " << point[1] << " | Nearest Y Point: " << nearest_point[1] << endl;
	cout << "Z input: " << point[2] << " | Nearest Z Point: " << nearest_point[2] << endl << endl;

	cout << "CASE WHERE DATA IS NOT ON GRID BUT IN GRID" << endl;
	point[0] = 121.; point[1] = 65.; point[2] = 252.3337;
	nearest_point[0] = -1.; nearest_point[1] = -1.; nearest_point[2] = -1.;
	_testNearestIndex(point, nearest_point);
	cout << "OUTPUT" << endl;
	cout << "X input: " << point[0] << " | Nearest X Point: " << nearest_point[0] << endl;
	cout << "Y input: " << point[1] << " | Nearest Y Point: " << nearest_point[1] << endl;
	cout << "Z input: " << point[2] << " | Nearest Z Point: " << nearest_point[2] << endl << endl;

	cout << "CASE WHERE DATA IS OUTSIDE OF THE DOMAIN" << endl;
	point[0] = -100.; point[1] = 10000.; point[2] = 5000.;
	nearest_point[0] = -1.; nearest_point[1] = -1.; nearest_point[2] = -1.;
	_testNearestIndex(point, nearest_point);
	cout << "OUTPUT" << endl;
	cout << "X input: " << point[0] << " | Nearest X Point: " << nearest_point[0] << endl;
	cout << "Y input: " << point[1] << " | Nearest Y Point: " << nearest_point[1] << endl;
	cout << "Z input: " << point[2] << " | Nearest Z Point: " << nearest_point[2] << endl << endl;
	cout << "END NEAREST GRIDPOINT TEST" << endl << endl << endl << endl;
}

// the CPU method for testing the GPU method that returns the trilinear interpolation
// weights for a regular 3D isotropic grid. 
void testCalcWeights() {
	float *point = new float[3];
	float *weights_out = new float[8];

	// sett all weights to -999.99 to make 
	// certain they're being copied over from
	// the GPU
	for (int i = 0; i < 8; i++) {
		weights_out[i] = -999.99;
	}

	// in this case, w1 should equal 1 and the other weights equal 0.
	cout << endl << "TESTING THE INTERPOLATION WEIGHTS CALCULATOR USING 30 METER ISOTROPIC" << endl;
	cout << "CASE WHERE DATA IS EXACTLY ON GRID" << endl;
	point[0] = 30.; point[1] = 30.; point[2] = 30.;
	_testCalcWeights(point, weights_out);
	cout << "OUTPUT" << endl;
	for (int i = 0; i < 8; i++) {
		cout << "W" << i+1 << "=" << weights_out[i] << " ";
	}
	cout << endl << endl;

	// in this case, we should have 8 distinct weights with values < 1
	cout << "CASE WHERE DATA IS NOT ON GRID BUT IN GRID" << endl;
	point[0] = 121.; point[1] = 65.; point[2] = 252.3337;
	_testCalcWeights(point, weights_out);
	cout << "OUTPUT" << endl;
	for (int i = 0; i < 8; i++) {
		cout << "W" << i+1 << "=" << weights_out[i] << " ";
	}
	cout << endl << endl;

	// in this case, all weights should return as -1 because 
	// the requested point was outside of our specified domain
	cout << "CASE WHERE DATA IS OUTSIDE OF THE DOMAIN" << endl;
	point[0] = -100.; point[1] = 10000.; point[2] = 5000.;
	_testCalcWeights(point, weights_out);
	cout << "OUTPUT" << endl;
	for (int i = 0; i < 8; i++) {
		cout << "W" << i+1 << "=" << weights_out[i] << " ";
	}
	cout << endl << endl;
	cout << "END CALC WEIGHTS TEST" << endl << endl << endl << endl;



}

// the CPU method for testing the GPU method that returns the trilinear interpolation
// weights for a regular 3D isotropic grid. 
void testTriInterp() {
	float *point = new float[3];
	float *interp_val = new float[1];

	// in this case, w1 should equal 1 and the other weights equal 0.
	cout << endl << "TESTING THE INTERPOLATION CALCULATOR USING 30 METER ISOTROPIC" << endl;
	cout << "CASE WHERE DATA IS EXACTLY ON GRID" << endl;
	point[0] = 30.; point[1] = 30.; point[2] = 60.;
	_testTriInterp(point, interp_val);
	cout << "OUTPUT" << endl;
	cout << "INTERP = " << interp_val[0];;
	cout << endl << endl;

	// in this case, we should have 8 distinct weights with values < 1
	cout << "CASE WHERE DATA IS NOT ON GRID BUT IN GRID" << endl;
	point[0] = 121.; point[1] = 65.; point[2] = 252.3337;
	_testTriInterp(point, interp_val);
	cout << "OUTPUT" << endl;
	cout << "INTERP = " << interp_val[0];;
	cout << endl << endl;

	// in this case, all weights should return as -1 because 
	// the requested point was outside of our specified domain
	cout << "CASE WHERE DATA IS OUTSIDE OF THE DOMAIN" << endl;
	point[0] = -100.; point[1] = 10000.; point[2] = 5000.;
	_testTriInterp(point, interp_val);
	cout << "OUTPUT" << endl;
	cout << "INTERP = " << interp_val[0];;
	cout << endl << endl;
	cout << "END CALC WEIGHTS TEST" << endl << endl << endl << endl;


}

int main() {

	testNearestIndex();
	testCalcWeights();
    testTriInterp();
	return 0;
}
