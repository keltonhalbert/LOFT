#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"
#include "readlofs.cpp"
#include "datastructs.cpp"
#include "integrate.h"

// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
#define P4(x,y,z,t,mx,my,mz) ((t*mx*my*mz)+((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
#define P2(t,p,mt) (((p)*(mt))+(t))
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
    requested_grid->X0 = saved_X0 + 80; requested_grid->Y0 = saved_Y0 + 40;
    requested_grid->X1 = saved_X0 + 280; requested_grid->Y1 = saved_Y0 + 240;
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

    lofs_read_3dvar(requested_grid, ubuffer, (char *)"uinterp", t0);
    lofs_read_3dvar(requested_grid, vbuffer, (char *)"vinterp", t0);
    lofs_read_3dvar(requested_grid, wbuffer, (char *)"winterp", t0);
}


/* Seed some test parcels into vectors of vectors for the different position dimension
 */
void seed_parcels(parcel_pos *parcels, datagrid *requested_grid) {

    int nParcels = parcels->nParcels;
    int pid = 0;
    for (int i = 30; i < 100; ++i) {
        for (int j = 100; j < 170; ++j) {
            parcels->xpos[P2(0, pid, parcels->nTimes)] = requested_grid->xh[i];
            parcels->ypos[P2(0, pid, parcels->nTimes)] = requested_grid->yh[j];
            parcels->zpos[P2(0, pid, parcels->nTimes)] = 1005.;
            pid += 1;
        }
    }

    for (int p = 0; p < nParcels; ++p) {
        for (int t = 1; t < parcels->nTimes; ++t) {
            parcels->xpos[P2(t, pid, parcels->nTimes)] = -99999.0;
            parcels->ypos[P2(t, pid, parcels->nTimes)] = -99999.0;
            parcels->zpos[P2(t, pid, parcels->nTimes)] = -99999.0;
        }
    }

}


/* Write out the parcel arrays to a 
 * CSV file format on the disk.
 */
void write_data(parcel_pos parcels, float *uparcels, float *vparcels, float *wparcels) {
    cout << "WRITING DATA" << endl;
    ofstream outfile;
    outfile.open("./result.csv");
    int nParcels = parcels.nParcels;
    int nT = parcels.nTimes;
    
    // loop over each parcel
    for (int pcl = 0; pcl < nParcels; ++pcl) {
        // print the parcel start flag 
        outfile << "!Parcel " << pcl << endl; 
        // loop over the times
        for (int t = 0; t < nT; ++t) {
            // for each row: x position, y position, z position
            for (int row = 0; row < 6; ++row) {
                if (row == 0) outfile << parcels.xpos[P2(t, pcl, nT)] << ", ";
                if (row == 1) outfile << parcels.ypos[P2(t, pcl, nT)] << ", ";
                if (row == 2) outfile << parcels.zpos[P2(t, pcl, nT)] << ", ";
                if (row == 3) outfile << uparcels[P2(t, pcl, nT)] << ", ";
                if (row == 4) outfile << vparcels[P2(t, pcl, nT)] << ", ";
                if (row == 5) outfile << wparcels[P2(t, pcl, nT)] << endl;
            }
        }
        // parcel end flag
        outfile << "!End " << pcl << endl;
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
    int rank, size;
    long N, MX, MY, MZ;
    int nTimeChunks = 1;

    // initialize a bunch of MPI stuff.
    // Rank tells you which process
    // you are and size tells y ou how
    // many processes there are total
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_ARE_FATAL); /* return info about
                                                                   errors */
    MPI_Barrier(MPI_COMM_WORLD);

    // the number of time steps we have is 
    // the number of MPI ranks there are
    // times the number of integration time chunks,
    // plus the very last integration end time
    int nTotTimes = (size*nTimeChunks)+1;

    // we're gonna make a test by creating a horizontal
    // and zonal line of parcels
    int nParcels = 4900;
    parcel_pos parcels;
    datagrid requested_grid;

    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    parcels.xpos = new float[nParcels * nTotTimes];
    parcels.ypos = new float[nParcels * nTotTimes];
    parcels.zpos = new float[nParcels * nTotTimes];
    parcels.nParcels = nParcels;
    parcels.nTimes = nTotTimes;

    for (int tChunk = 0; tChunk < nTimeChunks; ++tChunk) {

        // read in the metadata - later we will make
        // the requested grid dynamic based on the
        // parcel seeds
        loadMetadataAndGrid(base_dir, &requested_grid); 

        // if this is the first chunk of time, seed the
        // parcel start locations
        if (tChunk == 0) {
            cout << "SEEDING PARCELS" << endl;
            seed_parcels(&parcels, &requested_grid);
        }

        // the number of grid points requested
        N = (requested_grid.NX+1)*(requested_grid.NY+1)*(requested_grid.NZ+1);


        // get the size of the domain we will
        // be requesting. The +1 is safety for
        // staggered grids
        MX = (long) (requested_grid.NX+1);
        MY = (long) (requested_grid.NY+1);
        MZ = (long) (requested_grid.NZ+1);

        // allocate space for U, V, and W arrays
        long bufsize = (long) (requested_grid.NX+1) * (long) (requested_grid.NY+1) * (long) (requested_grid.NZ+1) * (long) sizeof(float);
        float *ubuf, *vbuf, *wbuf;
        ubuf = (float *) malloc ((size_t)bufsize);
        vbuf = (float *) malloc ((size_t)bufsize);
        wbuf = (float *) malloc ((size_t)bufsize);
        /*
        float *ubuf = new float[N];
        float *vbuf = new float[N];
        float *wbuf = new float[N];
        */

        // construct a 4D contiguous array to store stuff in.
        // bufsize is the size of the 3D component and size is
        // the number of MPI ranks (which is also the number of times)
        // read in
        float *u_time_chunk, *v_time_chunk, *w_time_chunk; 
        if (rank == 0) {
            u_time_chunk = (float *) malloc ((size_t)bufsize*size);
            v_time_chunk = (float *) malloc ((size_t)bufsize*size);
            w_time_chunk = (float *) malloc ((size_t)bufsize*size);
        }
        /*
        float *u_time_chunk = new float[N*size];
        float *v_time_chunk = new float[N*size];
        float *w_time_chunk = new float[N*size];
        */

        printf("TIMESTEP %d/%d %d %f\n", rank, size, rank + tChunk*size, alltimes[5]);
        // load u, v, and w into memory
        //loadVectorsFromDisk(&requested_grid, ubuf, vbuf, wbuf, alltimes[rank + tChunk*size]);
        
        // DEBUG: Let's have all ranks read in only 1 time, reducing any time variance.
        // Maybe this will help us track down our memory bug.
        loadVectorsFromDisk(&requested_grid, ubuf, vbuf, wbuf, alltimes[5]);


        int senderr_u = MPI_Gather(ubuf, N, MPI_FLOAT, u_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_v = MPI_Gather(vbuf, N, MPI_FLOAT, v_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_w = MPI_Gather(wbuf, N, MPI_FLOAT, w_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        cout << "GATHER ERROR CODE: " << senderr_u << " " << senderr_v << " " << senderr_w << endl;



        if (rank == 0) {
            // send to the GPU
            // comment out if you're running on XE node
            float *uparcels = new float[nParcels * nTotTimes];
            float *vparcels = new float[nParcels * nTotTimes];
            float *wparcels = new float[nParcels * nTotTimes];
            cudaIntegrateParcels(requested_grid, parcels, u_time_chunk, v_time_chunk, w_time_chunk, uparcels, vparcels, wparcels, MX, MY, MZ, size, tChunk, nTotTimes); 
            
            // if the last integration has been performed, write the data to disk
            if (tChunk == nTimeChunks-1) {
                write_data(parcels, uparcels, vparcels, wparcels);
            }
        }
    }

    MPI_Finalize();
}
