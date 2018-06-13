#include "mpi.h"
#include "readlofs.cpp"
#include "loadseeds.cpp"
#include <iostream>
#include <string>
#include <vector>
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
    long N;
    MPI_Status status;


    datagrid requested_grid;
    loadMetadataAndGrid(base_dir, &requested_grid);
    
    N = (requested_grid.NX+1)*(requested_grid.NY+1)*(requested_grid.NZ+1);

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int nT = size;
    long bufsize = (long) (requested_grid.NX+1) * (long) (requested_grid.NY+1) * (long) (requested_grid.NZ+1) * (long) sizeof(float);
    // allocate space for U, V, and W arrays
    float *ubuf = new float[(size_t)bufsize];
    float *vbuf = new float[(size_t)bufsize];
    float *wbuf = new float[(size_t)bufsize];
    if (rank == 0) {
        cout << endl << "GRID DIMENSIONS" << endl;
        cout << "NX = " << requested_grid.NX << " NY = " << requested_grid.NY << " NZ = " << requested_grid.NZ << endl;
        cout << "X0 = " << requested_grid.xh[requested_grid.X0] << " Y0 = " << requested_grid.yh[requested_grid.Y0] << endl;
        cout << "X1 = " << requested_grid.xh[requested_grid.X1] << " Y1 = " << requested_grid.yh[requested_grid.Y1] << endl;
        cout << "Z0 = " << requested_grid.zh[requested_grid.Z0] << " Z1 = " << requested_grid.zh[requested_grid.Z1] << endl;
        cout << "Allocating Memory: " << bufsize * 1.25e-7 << "mb per vector for " << endl;
    }


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
        float **u_time_chunk = new float*[nT];
        float **v_time_chunk = new float*[nT];
        float **w_time_chunk = new float*[nT];
        u_time_chunk[0] = ubuf;
        v_time_chunk[0] = vbuf;
        w_time_chunk[0] = wbuf;

        for (int i = 1; i < size; ++i) {
            u_time_chunk[i] = new float[(size_t)bufsize];
            v_time_chunk[i] = new float[(size_t)bufsize];
            w_time_chunk[i] = new float[(size_t)bufsize];
            MPI_Recv(u_time_chunk[i], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(v_time_chunk[i], N, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(w_time_chunk[i], N, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &status);
            cout << "Received from: " << status.MPI_SOURCE << " Error: " << status.MPI_ERROR << endl;
        }
    }

    MPI_Finalize();

}
