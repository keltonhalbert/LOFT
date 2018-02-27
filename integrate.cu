#include <iostream>
using namespace std;

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

// This is a GPU kernel for integrating a fluid parcel forward in time by increment dt.
// It expects the X, Y, and Z array pointers as well as the U, V, and W wind component pointers,
// which are all of length nParcels
__global__ void integrate(float *d_x_arr, float *d_y_arr, float *d_z_arr, float *d_u_arr, float *d_v_arr, float *d_w_arr, int nParcels, float dt) {
	// use the thread index to index the array
	int pidx = blockIdx.x*blockDim.x + threadIdx.x;

	// safety check to not access array memory out of bounds
	if (pidx < nParcels) {
		// integrate X position forward by the U wind
		d_x_arr[pidx] = d_x_arr[pidx] + d_u_arr[pidx] * dt;
		// integrate Y position forward by the V wind
		d_y_arr[pidx] = d_y_arr[pidx] + d_v_arr[pidx] * dt;
		// integrate Z position forward by the W wind
		d_z_arr[pidx] = d_z_arr[pidx] + d_w_arr[pidx] * dt;
	}
}


int main()
{
	//number of parcels to integrate
	int nParcels = 250000;
	// number of time steps to integrate
	int nTimes = 6*60*15;
	// integration time step (s)
	float dt = 1.0;

	// dynamically allocated 2D array for parcel trajectory on the (host) computer.
	// The first array dimension will be the number of time steps to integrate, and
	// the second array dimension will be the number of parcels nParcels
	float **x_traj_arr = new float*[nTimes];
	float **y_traj_arr = new float*[nTimes];
	float **z_traj_arr = new float*[nTimes];
	// array pointers used for the rows of the above arrays when 
	// iterating over the time dimension
	float *h_traj_row_x;
	float *h_traj_row_y;
	float *h_traj_row_z;
	// arrays of U,V,W used for integrating the parcels
	float *h_u_arr = new float[nParcels];
	float *h_v_arr = new float[nParcels];
	float *h_w_arr = new float[nParcels];

	// declare a pointer for GPU (device) memory
	// the device arrays for the row corresponding to
	// the current time during integration
	float *d_traj_row_x;
	float *d_traj_row_y;
	float *d_traj_row_z;
	// pointers to device memory for U,V,W arrays
	float *d_u_arr;
	float *d_v_arr;
	float *d_w_arr;

	// allocate the space on the GPU for the X,Y,Z positions
	gpuErrchk( cudaMallocManaged(&d_traj_row_x, nParcels * sizeof(float)) );
	gpuErrchk( cudaMallocManaged(&d_traj_row_y, nParcels * sizeof(float)) );
	gpuErrchk( cudaMallocManaged(&d_traj_row_z, nParcels * sizeof(float)) );

	// allocate the space on the GPU for the U, V, W arrays
	gpuErrchk( cudaMallocManaged(&d_u_arr, nParcels * sizeof(float)) );
	gpuErrchk( cudaMallocManaged(&d_v_arr, nParcels * sizeof(float)) );
	gpuErrchk( cudaMallocManaged(&d_w_arr, nParcels * sizeof(float)) );



	// finish dynamic allocation of (host) 2D arrays and initialize them with 0
	for (int t = 0; t < nTimes; t++) {
		x_traj_arr[t] = new float[nParcels];
		y_traj_arr[t] = new float[nParcels];
		z_traj_arr[t] = new float[nParcels];
		for (int p = 0; p < nParcels; p++) {
			x_traj_arr[t][p] = 0.;
			y_traj_arr[t][p] = 0.;
			z_traj_arr[t][p] = 0.;
		}
	}

	// allocate the 1D arrays for U,V,W
	for (int p = 0; p < nParcels; p++) {
		h_u_arr[p] = 1.;
		h_v_arr[p] = 1.;
		h_w_arr[p] = 1.;
	}


	for (int t = 0; t < nTimes - 1; t++) {
		// get the pointers to the current row of parcels
		// at this current time step t
		h_traj_row_x = x_traj_arr[t];
		h_traj_row_y = y_traj_arr[t];
		h_traj_row_z = z_traj_arr[t];

		// copy the X, Y, Z arrays to the device
		gpuErrchk( cudaMemcpy(d_traj_row_x, h_traj_row_x, nParcels * sizeof(float), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_traj_row_y, h_traj_row_y, nParcels * sizeof(float), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_traj_row_z, h_traj_row_z, nParcels * sizeof(float), cudaMemcpyHostToDevice) );

		// copy the U, V, W arrays to the device
		gpuErrchk( cudaMemcpy(d_u_arr, h_u_arr, nParcels * sizeof(float), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_v_arr, h_v_arr, nParcels * sizeof(float), cudaMemcpyHostToDevice) );
		gpuErrchk( cudaMemcpy(d_w_arr, h_w_arr, nParcels * sizeof(float), cudaMemcpyHostToDevice) );

		// run the (device) code on the GPU
		integrate<<<1024,1024>>>(d_traj_row_x, d_traj_row_y, d_traj_row_z, d_u_arr, d_v_arr, d_w_arr, nParcels, dt);
		gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );


		// copy the memory back from the GPU (device) to the computer (host)
		gpuErrchk( cudaMemcpy(x_traj_arr[t+1], d_traj_row_x, nParcels * sizeof(float), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(y_traj_arr[t+1], d_traj_row_y, nParcels * sizeof(float), cudaMemcpyDeviceToHost) );
		gpuErrchk( cudaMemcpy(z_traj_arr[t+1], d_traj_row_z, nParcels * sizeof(float), cudaMemcpyDeviceToHost) );

	}
	//cout << cudaMemGetInfo() << endl;


	cout << x_traj_arr[nTimes - 1][ nParcels - 1] << endl;
	cout << y_traj_arr[nTimes - 1][ nParcels - 1] << endl;
	cout << z_traj_arr[nTimes - 1][ nParcels - 1] << endl;
	// demonstrate the (host) array has been modified
	/*
	cout << endl;
	for (int t = 1; t < nTimes; t++) {
		for (int p = 0; p < nParcels; p++) {
			cout << traj_arr[t][p] << " ";
		}
		cout << endl;
	}
	*/

	cudaFree(d_traj_row_x);
	cudaFree(d_traj_row_y);
	cudaFree(d_traj_row_z);
	cudaFree(d_u_arr);
	cudaFree(d_v_arr);
	cudaFree(d_w_arr);




	return 0;
}