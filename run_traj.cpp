#include "mpi.h"
#include "readlofs.cpp"
#include "loadseeds.cpp"
#include <iostream>
#include <string>
#include <vector>
// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
#define P4(x,y,z,t,mx,my,mz) ((t*mx*my*mz)+((z)*(mx)*(my))+((y)*(mx))+(x))
using namespace std;

/* Load the grid metadata and get the gird based on requested bounds.
 * This only needs to be called once to load data into memory. Additionally,
 * subsequent calls will look for cached metadata in .cm1hdf5_* files.
 */
void loadMetadataAndGrid(string base_dir, datagrid *requested_grid) {

    // query the dataset structure
    lofs_get_dataset_structure(base_dir);
    // get the HDF metadata - return the first filename
    get_hdf_metadata(firstfilename,&nx,&ny,&nz,&nodex,&nodey);

    // for right now, set the grid bounds to the saved
    // bounds for testing purposes
    requested_grid->X0 = saved_X0; requested_grid->Y0 = saved_Y0 + 260;
    requested_grid->X1 = saved_X0 + 330; requested_grid->Y1 = saved_Y0 + 590;
    requested_grid->Z0 = 0; requested_grid->Z1 = 100;

    // request a grid subset based on 
    // the subset information provided to
    // out grid struct
    lofs_get_grid(requested_grid);
}

/* Read in the U, V, and W vector components from the disk, provided previously allocated memory buffers
 * and the time requested in the dataset. 
 */
void loadVectorsFromDisk(datagrid *requested_grid, float *ubuffer, float *vbuffer, float *wbuffer, double t0) {
    // request 3D field!
    lofs_read_3dvar(requested_grid, ubuffer, (char *)"u", t0);
    lofs_read_3dvar(requested_grid, vbuffer, (char *)"v", t0);
    lofs_read_3dvar(requested_grid, wbuffer, (char *)"w", t0);
}

int main() {
    string base_dir = "/u/sciteam/halbert/project_bagm/khalbert/30m-every-time-step/3D";
    double t0 = 3001.;
    int rank, size;
    long N, MX, MY, MZ;
    MPI_Status status;


    // read in the metadata
    datagrid requested_grid;
    loadMetadataAndGrid(base_dir, &requested_grid);
    
    // the number of grid points requested
    N = (requested_grid.NX+1)*(requested_grid.NY+1)*(requested_grid.NZ+1);

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int nT = size;
    MX = (long) (requested_grid.NX+1);
    MY = (long) (requested_grid.NY+1);
    MZ = (long) (requested_grid.NZ+1);
    long bufsize = MX * MY * MZ * (long) sizeof(float);
    if (rank == 0) {
        cout << endl << "GRID DIMENSIONS" << endl;
        cout << "NX = " << requested_grid.NX << " NY = " << requested_grid.NY << " NZ = " << requested_grid.NZ << endl;
        cout << "X0 = " << requested_grid.xh[requested_grid.X0] << " Y0 = " << requested_grid.yh[requested_grid.Y0] << endl;
        cout << "X1 = " << requested_grid.xh[requested_grid.X1] << " Y1 = " << requested_grid.yh[requested_grid.Y1] << endl;
        cout << "Z0 = " << requested_grid.zh[requested_grid.Z0] << " Z1 = " << requested_grid.zh[requested_grid.Z1] << endl;
        cout << "Allocating Memory: " << bufsize * 1.25e-7 << "mb per vector for " << endl;
    }


    // allocate space for U, V, and W arrays
    float *ubuf = new float[(size_t)bufsize];
    float *vbuf = new float[(size_t)bufsize];
    float *wbuf = new float[(size_t)bufsize];
    cout << "TIMESTEP " << rank << " " << alltimes[rank] <<  endl;
    loadVectorsFromDisk(&requested_grid, ubuf, vbuf, wbuf, alltimes[rank]);



    if (rank != 0) {
        int dest = 0;
        cout << "Sending from: " << rank << endl;
        MPI_Send(ubuf, N, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
        MPI_Send(vbuf, N, MPI_FLOAT, dest, 2, MPI_COMM_WORLD);
        MPI_Send(wbuf, N, MPI_FLOAT, dest, 3, MPI_COMM_WORLD);
        delete[] wbuf;
        delete[] ubuf;
        delete[] vbuf;

    }

    else {
        // construct a 4D contiguous array to store stuff in.
        // bufsize is the size of the 3D component and size is
        // the number of MPI ranks (which is also the number of times)
        // read in
        float *u_time_chunk = new float[(size_t)(bufsize*size)];
        float *v_time_chunk = new float[(size_t)(bufsize*size)];
        float *w_time_chunk = new float[(size_t)(bufsize*size)];

        // we need to add the buffered data to the 4D array
        // for our rank (rank 0)
        for (int idx = 0; idx < N; ++idx) {
            // obviously since it's time 0 this doesn't matter
            // but I'm showing it for clarity
            u_time_chunk[0*MX*MY*MZ + idx] = ubuf[idx];
            v_time_chunk[0*MX*MY*MZ + idx] = vbuf[idx];
            w_time_chunk[0*MX*MY*MZ + idx] = wbuf[idx];
        }

        // loop over the MPI ranks and receive the data 
        // transmitted from each rank
        for (int i = 1; i < size; ++i) {
            // get the buffers from the other MPI ranks
            // and place it into our 4D array at the corresponding time
            MPI_Recv(&(u_time_chunk[i*MX*MY*MZ]), N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&(v_time_chunk[i*MX*MY*MZ]), N, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&(w_time_chunk[i*MX*MY*MZ]), N, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &status);
            // report the status just in case
            cout << "Received from: " << status.MPI_SOURCE << " Error: " << status.MPI_ERROR << endl;
        }
        float max = -999.0;
        for (int i = 0; i < N*size; ++i) {
            if (w_time_chunk[i] > max) max = w_time_chunk[i];
        }
        cout << "Max is: " << max << endl;

    }

    MPI_Finalize();

}
