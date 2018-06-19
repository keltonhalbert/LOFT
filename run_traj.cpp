#include "mpi.h"
#include "datastructs.cpp"
#include "readlofs.cpp"
#include "loadseeds.cpp"
#include <iostream>
#include <string>
#include <vector>
#include "integrate.h"
// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
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
    requested_grid->X0 = saved_X0 + 180; requested_grid->Y0 = saved_Y0 + 180;
    requested_grid->X1 = saved_X0 + 380; requested_grid->Y1 = saved_Y0 + 380;
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


/* Seed some test parcels into vectors of vectors for the different position dimension
 */
void seed_parcels(parcel_pos *parcels, datagrid *requested_grid, int nParcels) {

    int pid = 0;
    for (int i = 30; i < 130; ++i) {
        for (int j = 100; j < 200; ++j) {
            parcels->xpos[0 + (parcels->nTimes*pid)] = requested_grid->xh[i];
            parcels->ypos[0 + (parcels->nTimes*pid)] = requested_grid->yh[j];
            parcels->zpos[0 + (parcels->nTimes*pid)] = 1005.;
            cout << "PID: " << pid << endl;
            pid += 1;
        }
    }

    for (int p = 0; p < nParcels; ++p) {
        for (int t = 1; t < parcels->nTimes; ++t) {
            parcels->xpos[t + (parcels->nTimes*p)] = -99999.0;
            parcels->ypos[t + (parcels->nTimes*p)] = -99999.0;
            parcels->zpos[t + (parcels->nTimes*p)] = -99999.0;
        }
    }

}


/* This is the main program that does the parcel trajectory analysis.
 * It first sets up the parcel vectors and seeds the starting locations.
 * It then loads a chunk of times into memory by calling the LOFS api
 * wrappers, with the number of times read in being determined by the
 * number of MPI ranks launched. It then passes the vectors and the 4D u/v/w 
 * data chunks to the GPU, and then proceeds with another time chunk.
 */
int main(int argc, char **argv ) {
    string base_dir = "/u/sciteam/halbert/project_bagm/khalbert/30m-every-time-step/3D";
    int rank, size, ierr_u, ierr_v, ierr_w, errclass;
    long N, MX, MY, MZ;
    MPI_Status status;

    // initialize a bunch of MPI stuff.
    // Rank tells you which process
    // you are and size tells y ou how
    // many processes there are total
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN); /* return info about
                                                                   errors */
    // the number of time steps we have is 
    // the number of MPI ranks there are
    // plus the last integration time
    int nT = size+1;


    // read in the metadata
    datagrid requested_grid;
    loadMetadataAndGrid(base_dir, &requested_grid);

    
    // the number of grid points requested
    N = (requested_grid.NX+1)*(requested_grid.NY+1)*(requested_grid.NZ+1);


    // get the size of the domain we will
    // be requesting. The +1 is safety for
    // staggered grids
    MX = (long) (requested_grid.NX+1);
    MY = (long) (requested_grid.NY+1);
    MZ = (long) (requested_grid.NZ+1);

    // the size of our 3D buffer
    long bufsize = MX * MY * MZ * (long) sizeof(float);

    // have the master rank print out some diagrostic information
    if (rank == 0) {
        cout << endl << "GRID DIMENSIONS" << endl;
        cout << "NX = " << requested_grid.NX << " NY = " << requested_grid.NY << " NZ = " << requested_grid.NZ << endl;
        cout << "X0 = " << requested_grid.xh[requested_grid.X0] << " Y0 = " << requested_grid.yh[requested_grid.Y0] << endl;
        cout << "X1 = " << requested_grid.xh[requested_grid.X1] << " Y1 = " << requested_grid.yh[requested_grid.Y1] << endl;
        cout << "Z0 = " << requested_grid.zh[requested_grid.Z0] << " Z1 = " << requested_grid.zh[requested_grid.Z1] << endl;
        cout << "Allocating Memory: " << bufsize * 1.25e-7 << "mb per vector for " << endl;
    }


    // allocate space for U, V, and W arrays
    float *ubuf = new float[N];
    float *vbuf = new float[N];
    float *wbuf = new float[N];
    cout << "TIMESTEP " << rank << " " << alltimes[rank] <<  endl;
    // load u, v, and w into memory
    loadVectorsFromDisk(&requested_grid, ubuf, vbuf, wbuf, alltimes[rank]);


    // if this is nor the master rank, communicate
    // the data to the master rank and then delete
    // the buffers since only rank 0 is used from
    // here on out
    if (rank != 0) {
        int dest = 0;
        cout << "Sending from: " << rank << endl;
        // send the U, V, and W arrays to the destination rank (0)
        // and use tag 1 for U, tag 2 for V, and tag 3 for W
        ierr_u = MPI_Send(ubuf, N, MPI_FLOAT, dest, 1, MPI_COMM_WORLD);
        ierr_v = MPI_Send(vbuf, N, MPI_FLOAT, dest, 2, MPI_COMM_WORLD);
        ierr_w = MPI_Send(wbuf, N, MPI_FLOAT, dest, 3, MPI_COMM_WORLD);

        // lets do some error handling just in case
        if ((ierr_u != MPI_SUCCESS) || (ierr_v != MPI_SUCCESS) || (ierr_w != MPI_SUCCESS)) {
            cout << "MPI Communication Error on rank: " << rank << endl;
            cout << "U array error status: " << ierr_u << endl;
            cout << "V array error status: " << ierr_v << endl;
            cout << "W array error status: " << ierr_w << endl;
        }
        if ((ierr_u == MPI_SUCCESS) && (ierr_v == MPI_SUCCESS) && (ierr_w == MPI_SUCCESS)) {
            cout << "Succeffully passed U/V/W arrays from rank: " << rank << endl;
        }
        // de-allocate the memory the rank uses after communicating
        delete[] wbuf;
        delete[] ubuf;
        delete[] vbuf;
    }

    // If this is the master rank (rank == 0)
    // then do all the cool stuff
    else {
        // construct a 4D contiguous array to store stuff in.
        // bufsize is the size of the 3D component and size is
        // the number of MPI ranks (which is also the number of times)
        // read in
        float *u_time_chunk = new float[N*size];
        float *v_time_chunk = new float[N*size];
        float *w_time_chunk = new float[N*size];

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
            cout << "Received from: " << status.MPI_SOURCE << " Error: " << status.MPI_ERROR << endl;
        }

        // we're gonna make a test by creating a horizontal
        // and zonal line of parcels
        int nParcels = 10000;
        parcel_pos parcels;
        float *xpos = new float[nParcels * nT];
        float *ypos = new float[nParcels * nT];
        float *zpos = new float[nParcels * nT];
        parcels.xpos = xpos; parcels.ypos = ypos;
        parcels.zpos = zpos;
        parcels.nParcels = nParcels;
        parcels.nTimes = nT;

        // seed the parcels
        seed_parcels(&parcels, &requested_grid, nParcels);
        // print to make sure we properly seeded
        for (int i = 0; i < nParcels; ++i) {
            // sanity print to make sure we're seeding the right stuff
            cout << "Starting Positions: X = " << parcels.xpos[nT*i] << " Y = " << parcels.ypos[nT*i] << " Z = " << parcels.zpos[nT*i] << endl;
        }

        cudaIntegrateParcels(requested_grid, parcels, u_time_chunk, v_time_chunk, w_time_chunk, MX, MY, MZ, nT);
    }

    MPI_Finalize();

}
