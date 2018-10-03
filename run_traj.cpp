#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"
#include "datastructs.cpp"
#include "readlofs.cpp"
#include "integrate.h"
#include "writenc.cpp"

// I made this myself by stealing from LOFS
#define P2(t,p,mt) (((p)*(mt))+(t))
// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
#define P4(x,y,z,t,mx,my,mz) ((t*mx*my*mz)+((z)*(mx)*(my))+((y)*(mx))+(x))
using namespace std;


// this was stolen from LOFS/cm1tools-3.0 hdf2.c
// under the parce_cmdline_hdf2nc function. I could
// have just linked to it, but I want the LOFS dependency
// to only be I/O so that other backends can be used, hence
// copying it here.
void parse_cmdline(int argc, char **argv, \
	char *histpath, char *base, double *time, \
	double *X0, double *Y0, double *Z0, int *NX, int *NY, int *NZ, \
    double *DX, double *DY, double *DZ )
{
	int got_histpath,got_base,got_time,got_X0,got_NX,got_Y0,got_NY,got_Z0,got_NZ,got_DX,got_DY,got_DZ;
    int argc_min=1;
    int optcount=0;
	enum { OPT_HISTPATH = 1000, OPT_BASE, OPT_TIME, OPT_X0, OPT_Y0, OPT_Z0, OPT_NX, OPT_NY, OPT_NZ, OPT_DX,
		OPT_DY, OPT_DZ, OPT_DEBUG, OPT_XYF, OPT_YES2D, OPT_NC3, OPT_COMPRESS, OPT_NTHREADS };
	// see https://stackoverflow.com/questions/23758570/c-getopt-long-only-without-alias
	static struct option long_options[] =
	{
		{"histpath", required_argument, 0, OPT_HISTPATH},
		{"base",     required_argument, 0, OPT_BASE},
		{"time",     required_argument, 0, OPT_TIME},
		{"x0",       required_argument, 0, OPT_X0},
		{"y0",       required_argument, 0, OPT_Y0},
		{"z0",       required_argument, 0, OPT_Z0},
		{"nx",       required_argument, 0, OPT_NX},
		{"ny",       required_argument, 0, OPT_NY},
		{"nz",       required_argument, 0, OPT_NZ},
		{"dx",       required_argument, 0, OPT_DX},
		{"dy",       required_argument, 0, OPT_DY},
		{"dz",       required_argument, 0, OPT_DZ},
		{"debug",    optional_argument, 0, OPT_DEBUG},
		{0, 0, 0, 0}//sentinel, needed!
	};

	got_histpath=got_base=got_time=got_X0=got_NX=got_DX=got_Y0=got_NY=got_DY=got_Z0=got_NZ=got_DZ=0;

	int bail = 0;

	if (argc == 1)
	{
		fprintf(stderr,
		"Usage: %s --histpath=[histpath] --base=[base] --x0=[X0] --y0=[Y0] --z0=[Z0] --nx=[NX] --ny=[NY] --nz=[NZ] --dx=[DX] --dy=[DY] --dz=[DZ] --time=[time] [varname1 ... varnameN] \n",argv[0]);
		exit(0);
	}

	while (1)
	{
		int r;
		int option_index = 0;
		r = getopt_long_only (argc, argv,"",long_options,&option_index);
		if (r == -1) break;

		switch(r)
		{
			case OPT_HISTPATH:
				strcpy(histpath,optarg);
				got_histpath=1;
				printf("histpath = %s\n",histpath);
				break;
			case OPT_BASE:
				strcpy(base,optarg);
				got_base=1;
				printf("base = %s\n",base);
				break;
			case OPT_TIME:
				*time = atof(optarg);
				got_time=1;
				printf("parcel start time time = %f\n",*time);
				break;
			case OPT_X0:
				*X0 = atof(optarg);
				got_X0=1;
				optcount++;
				printf("parcel seed start X0 = %f\n",*X0);
				break;
			case OPT_Y0:
				*Y0 = atof(optarg);
				got_Y0=1;
				optcount++;
				printf("parcel seed start Y0 = %f\n",*Y0);
				break;
			case OPT_Z0:
				*Z0 = atof(optarg);
				got_Z0=1;
				optcount++;
				printf("parcel seed start Z0 = %f\n",*Z0);
				break;
			case OPT_NX:
				*NX = atoi(optarg);
				got_NX=1;
				optcount++;
				printf("number of parcels NX = %i\n",*NX);
				break;
			case OPT_NY:
				*NY = atoi(optarg);
				got_NY=1;
				optcount++;
				printf("number of parcels NY = %i\n",*NY);
				break;
			case OPT_NZ:
				*NZ = atoi(optarg);
				got_NZ=1;
				optcount++;
				printf("number of parcels NZ = %i\n",*NZ);
				break;
			case OPT_DX:
				*DX = atof(optarg);
				got_DX=1;
				optcount++;
				printf("spacing of parcels DX = %f\n",*DX);
				break;
			case OPT_DY:
				*DY = atof(optarg);
				got_DY=1;
				optcount++;
				printf("spacing of parcels DY = %f\n",*DY);
				break;
			case OPT_DZ:
				*DZ = atof(optarg);
				got_DZ=1;
				optcount++;
				printf("spacing of parcels DZ = %f\n",*DZ);
				break;
			case OPT_DEBUG:
				debug=1;
				optcount++;
				break;
			case '?':
				fprintf(stderr,"Exiting: unknown command line option.\n");
				exit(0);
				break;
		}
	}

		if (got_histpath==0) { fprintf(stderr,"--histpath not specified\n"); bail = 1; }
		if (got_base==0)   { fprintf(stderr,"--base not specified\n"); bail = 1; }
		if (got_time==0)   { fprintf(stderr,"--time not specified\n"); bail = 1; }

/* These are now optional */
		if (!got_X0)      fprintf(stderr,"--x0 not specified; where do you want your parcels?\n");
		if (!got_Y0)      fprintf(stderr,"--y0 not specified; where do you want your parcels?\n");
		if (!got_Z0)      fprintf(stderr,"--z0 not specified; where do you want your parcels?\n");
		if (!got_NX)      fprintf(stderr,"--nx not specified; how many parcels do you want?\n");
		if (!got_NY)      fprintf(stderr,"--ny not specified; how many parcels do you want?\n");
		if (!got_NZ)      fprintf(stderr,"--nz not specified; how many parcels do you want?\n");
		if (!got_DX)      fprintf(stderr,"--dx not specified; how dense do you want your parcels?\n");
		if (!got_DY)      fprintf(stderr,"--dy not specified; how dense do you want your parcels?\n");
		if (!got_DZ)      fprintf(stderr,"--dz not specified; how dense do you want your parcels?\n");

		if (bail)           { fprintf(stderr,"Insufficient arguments to %s, exiting.\n",argv[0]); exit(-1); }
}

/* Load the grid metadata and request a domain subset based on the 
 * current parcel positioning for the current time step. The idea is that 
 * for the first chunk of times read in (from 0 to N MPI ranks for time)
 * only the subset of the domain that matters for that period of time. 
 * When the next chunk of time is read in, check and see where the parcels
 * are currently and request a subset that is relevent to those parcels.   
 */
void loadMetadataAndGrid(string base_dir, datagrid *requested_grid, parcel_pos *parcels) {
    // get the HDF metadata - return the first filename
    get_hdf_metadata(firstfilename,&nx,&ny,&nz,&nodex,&nodey);

    datagrid temp_grid;

    // for right now, set the grid bounds to the saved
    // bounds for testing purposes
    temp_grid.X0 = saved_X0; temp_grid.Y0 = saved_Y0;
    temp_grid.X1 = saved_X1; temp_grid.Y1 = saved_Y1;
    temp_grid.Z0 = 0; temp_grid.Z1 = nz-1; // nz comes from readlofs

    // request the full grid so that we can find the indices
    // of where our parcels are, and then request a smaller
    // subset from there.
    lofs_get_grid(&temp_grid);

    // find the min/max index bounds of 
    // our parcels
    float point[3];
    int idx_4D[4];
    int min_i = temp_grid.NX+1;
    int min_j = temp_grid.NY+1;
    int min_k = temp_grid.NZ+1;
    int max_i = -1;
    int max_j = -1;
    int max_k = -1;
    int invalidCount = 0;
    for (int pcl = 0; pcl < parcels->nParcels; ++pcl) {
        point[0] = parcels->xpos[P2(0, pcl, parcels->nTimes)];
        point[1] = parcels->ypos[P2(0, pcl, parcels->nTimes)];
        point[2] = parcels->zpos[P2(0, pcl, parcels->nTimes)];
        // find the nearest grid point!
        _nearest_grid_idx(point, temp_grid, idx_4D, temp_grid.NX, temp_grid.NY, temp_grid.NZ);
        if ( (idx_4D[0] == -1) || (idx_4D[1] == -1) || (idx_4D[2] == -1) ) {
            cout << "INVALID POINT X " << point[0] << " Y " << point[1] << " Z " << point[2] << endl;
            cout << "Parcel X " << parcels->xpos[P2(0, pcl, parcels->nTimes)];
            cout << " Parcel Y " << parcels->ypos[P2(0, pcl, parcels->nTimes)];
            cout << " Parcel Z " << parcels->zpos[P2(0, pcl, parcels->nTimes)] << endl;
            invalidCount += 1;
        }

        // check to see if we've found the min/max
        // for the dimension
        if (idx_4D[0] < min_i) min_i = idx_4D[0]; 
        if (idx_4D[0] > max_i) max_i = idx_4D[0]; 
        if (idx_4D[1] < min_j) min_j = idx_4D[1]; 
        if (idx_4D[1] > max_j) max_j = idx_4D[1]; 
        if (idx_4D[2] < min_k) min_k = idx_4D[2]; 
        if (idx_4D[2] > max_k) max_k = idx_4D[2]; 
        //cout << "min i " << min_i << " max i " << max_i << endl;
        //cout << "min j " << min_j << " max i " << max_j << endl;
        //cout << "min k " << min_k << " max i " << max_k << endl;
    }
    requested_grid->isValid = 1;
    // clear the memory from the temp grid
    delete[] temp_grid.xh;
    delete[] temp_grid.yh;
    delete[] temp_grid.zh;
    delete[] temp_grid.xf;
    delete[] temp_grid.yf;
    delete[] temp_grid.zf;
    if (invalidCount == parcels->nParcels) {
        requested_grid->isValid = 0;
        return;
    }


    // we want to add a buffer to our dimensions so that
    // the parcels don't accidentally move outside of our
    // requested data. If the buffer goes outside the 
    // saved dimensions, set it to the saved dimensions
    min_i = saved_X0 + min_i - 5;
    max_i = saved_X0 + max_i + 5;
    min_j = saved_Y0 + min_j - 5;
    max_j = saved_Y0 + max_j + 5;
    min_k = min_k - 5;
    max_k = max_k + 5;
    cout << "Attempted Parcel Bounds In Grid" << endl;
    cout << "X0: " << min_i << " X1: " << max_i << endl;
    cout << "Y0: " << min_j << " Y1: " << max_j << endl;
    cout << "Z0: " << min_k << " Z1: " << max_k << endl;

    // keep the data in our saved bounds
    if (min_i < saved_X0) min_i = saved_X0;
    if (max_i > saved_X1) max_i = saved_X1;
    if (min_j < saved_Y0) min_j = saved_Y0;
    if (max_j > saved_Y1) max_j = saved_Y1;
    if (min_k < 0) min_k = 0;
    if (max_k > nz-1) max_k = nz-1;



    cout << "Parcel Bounds In Grid" << endl;
    cout << "X0: " << min_i << " X1: " << max_i << endl;
    cout << "Y0: " << min_j << " Y1: " << max_j << endl;
    cout << "Z0: " << min_k << " Z1: " << max_k << endl;


    requested_grid->X0 = min_i; requested_grid->Y0 = min_j;
    requested_grid->X1 = max_i; requested_grid->Y1 = max_j;
    requested_grid->Z0 = min_k; requested_grid->Z1 = max_k;

    // request our grid subset now
    cout << "REQUESTING METADATA & GRID" << endl;
    lofs_get_grid(requested_grid);
    cout << "END METADATA & GRID REQUEST" << endl;

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


/* Seed some parcels into the domain
 * in physical gridpoint space, and then
 * fill the remainder of the parcel traces
 * with missing values. 
 */
void seed_parcels(parcel_pos *parcels, int nTotTimes) {
    // place a cube of parcels in the domain between xstart, 
    // xend, ystart, yend, zstart, and zend in spacing
    // increments dx, dy, dz
    float xstart = -3775; float xend = -2225; float dx = 30;
    float ystart = -3775; float yend = -3225; float dy = 30;
    float zstart = 120; float zend = 330; float dz = 30;
    // the number of parcels we will be seeding
    // - note I kind of made this by trial and error. 
    // It may be subject to seg faults?
    int pnx = (int) ceil((xend - xstart) / dx);
    int pny = (int) ceil((yend - ystart) / dy);
    int pnz = (int) ceil((zend - zstart) / dz);
    int nParcels = pnx*pny*pnz;

    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    parcels->xpos = new float[nParcels * nTotTimes];
    parcels->ypos = new float[nParcels * nTotTimes];
    parcels->zpos = new float[nParcels * nTotTimes];
    parcels->pclu = new float[nParcels * nTotTimes];
    parcels->pclv = new float[nParcels * nTotTimes];
    parcels->pclw = new float[nParcels * nTotTimes];
    parcels->nParcels = nParcels;
    parcels->nTimes = nTotTimes;

    int pid = 0;
    for (float i = xstart; i < xend; i+=dx) {
        for (float j = ystart; j < yend; j+=dy) {
            for (float k = zstart; k < zend; k += dz) {
                parcels->xpos[P2(0, pid, parcels->nTimes)] = i;
                parcels->ypos[P2(0, pid, parcels->nTimes)] = j;
                parcels->zpos[P2(0, pid, parcels->nTimes)] = k;
                pid += 1;
            }
        }
    }

    // fill the remaining portions of the array
    // with the missing value flag for the future
    // times that we haven't integrated to yet.
    cout <<  parcels->zpos[P2(0, 29, parcels->nTimes)] << endl;
    for (int p = 0; p < nParcels; ++p) {
        for (int t = 1; t < parcels->nTimes; ++t) {
            parcels->xpos[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->ypos[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->zpos[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
        }
    }
    cout << "END PARCEL SEED" << endl;

}


void seed_parcels_cm1(parcel_pos *parcels, int nTotTimes) {

    int nParcels = 36000;

    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    parcels->xpos = new float[nParcels * nTotTimes];
    parcels->ypos = new float[nParcels * nTotTimes];
    parcels->zpos = new float[nParcels * nTotTimes];
    parcels->pclu = new float[nParcels * nTotTimes];
    parcels->pclv = new float[nParcels * nTotTimes];
    parcels->pclw = new float[nParcels * nTotTimes];

    parcels->nParcels = nParcels;
    parcels->nTimes = nTotTimes;

    int pid = 0;
    for (int k = 0; k < 10; ++k) {
        for (int j = 0; j < 60; ++j) {
            for (int i = 0; i < 60; ++i) {
                parcels->xpos[P2(0, pid, parcels->nTimes)] = -6375 + 250.0*i;
                parcels->ypos[P2(0, pid, parcels->nTimes)] = 5125 + 250.0*j;
                parcels->zpos[P2(0, pid, parcels->nTimes)] = 100 + 250.0*k;
                pid += 1;
            }
        }
    }

    // fill the remaining portions of the array
    // with the missing value flag for the future
    // times that we haven't integrated to yet.
    for (int p = 0; p < nParcels; ++p) {
        for (int t = 1; t < parcels->nTimes; ++t) {
            parcels->xpos[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->ypos[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->zpos[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->pclu[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->pclv[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->pclw[P2(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
        }
    }
    cout << "END PARCEL SEED" << endl;
}


/* This is the main program that does the parcel trajectory analysis.
 * It first sets up the parcel vectors and seeds the starting locations.
 * It then loads a chunk of times into memory by calling the LOFS api
 * wrappers, with the number of times read in being determined by the
 * number of MPI ranks launched. It then passes the vectors and the 4D u/v/w 
 * data chunks to the GPU, and then proceeds with another time chunk.
 */
int main(int argc, char **argv ) {
    double X0, DX, Y0, DY, Z0, DZ;
    int NX, NY, NZ;
    double time;
    char *base = new char[256];
    char *histpath = new char[256];
    parse_cmdline(argc, argv, histpath, base, &time, &X0, &Y0, &Z0, &NX, &NY, &NZ, &DX, &DY, &DZ );
    cout << "I'm leaving the command line parse and exiting" << endl;
    exit(-1);
    string base_dir = "/iliad/orfstore/khalbert/history.24May_r16_fixed-200-a/3D";
    string outfilename = "cuda-parcel.nc";
    // query the dataset structure
    int rank, size;
    long N, MX, MY, MZ;
    int nTimeChunks = 120*2;

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
    // plus the very last integration end time
    int nTotTimes = size+1;
    lofs_get_dataset_structure(base_dir);

    parcel_pos parcels;
    datagrid requested_grid;


    for (int tChunk = 0; tChunk < nTimeChunks; ++tChunk) {
        // if this is the first chunk of time, seed the
        // parcel start locations
        if (tChunk == 0) {
            cout << "SEEDING PARCELS" << endl;
            seed_parcels_cm1(&parcels, nTotTimes);
            if (rank == 0) init_nc(outfilename, &parcels);
        }

        // read in the metadata - later we will make
        // the requested grid dynamic based on the
        // parcel seeds
        loadMetadataAndGrid(base_dir, &requested_grid, &parcels); 
        if (requested_grid.isValid == 0) continue;


        // the number of grid points requested
        N = (requested_grid.NX)*(requested_grid.NY)*(requested_grid.NZ);


        // get the size of the domain we will
        // be requesting. The +1 is safety for
        // staggered grids
        MX = (long) (requested_grid.NX);
        MY = (long) (requested_grid.NY);
        MZ = (long) (requested_grid.NZ);

        // allocate space for U, V, and W arrays
        long bufsize = (long) (requested_grid.NX+1) * (long) (requested_grid.NY+1) * (long) (requested_grid.NZ+1) * (long) sizeof(float);
        float *ubuf, *vbuf, *wbuf;
        ubuf = new float[(size_t)bufsize];
        vbuf = new float[(size_t)bufsize];
        wbuf = new float[(size_t)bufsize];
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
        printf("TIMESTEP %d/%d %d %f\n", rank, size, rank + tChunk*size, alltimes[4199 + rank + tChunk*size]);
        // load u, v, and w into memory
        loadVectorsFromDisk(&requested_grid, ubuf, vbuf, wbuf, alltimes[4199 + rank + tChunk*size]);
        
        int senderr_u = MPI_Gather(ubuf, N, MPI_FLOAT, u_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_v = MPI_Gather(vbuf, N, MPI_FLOAT, v_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_w = MPI_Gather(wbuf, N, MPI_FLOAT, w_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);


        if (rank == 0) {
            // send to the GPU
            // comment out if you're running on XE node
            int nParcels = parcels.nParcels;
            cudaIntegrateParcels(requested_grid, parcels, u_time_chunk, v_time_chunk, w_time_chunk, MX, MY, MZ, size, nTotTimes); 
            write_parcels(outfilename, &parcels, tChunk);

            for (int pcl = 0; pcl < parcels.nParcels; ++pcl) {
                parcels.xpos[P2(0, pcl, parcels.nTimes)] = parcels.xpos[P2(size, pcl, parcels.nTimes)];
                parcels.ypos[P2(0, pcl, parcels.nTimes)] = parcels.ypos[P2(size, pcl, parcels.nTimes)];
                parcels.zpos[P2(0, pcl, parcels.nTimes)] = parcels.zpos[P2(size, pcl, parcels.nTimes)];
                parcels.pclu[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclv[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclw[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                for (int t = 1; t < parcels.nTimes; ++t) {
                    parcels.xpos[P2(t, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                    parcels.ypos[P2(t, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                    parcels.zpos[P2(t, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                    parcels.pclu[P2(t, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                    parcels.pclv[P2(t, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                    parcels.pclw[P2(t, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                }
            }

            // communicate the data to the other ranks
            for (int r = 1; r < size; ++r) {
                 MPI_Send(parcels.xpos, nParcels*nTotTimes, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
                 MPI_Send(parcels.ypos, nParcels*nTotTimes, MPI_FLOAT, r, 1, MPI_COMM_WORLD);
                 MPI_Send(parcels.zpos, nParcels*nTotTimes, MPI_FLOAT, r, 2, MPI_COMM_WORLD);
             }



            delete[] requested_grid.xf;
            delete[] requested_grid.yf;
            delete[] requested_grid.zf;

            delete[] requested_grid.xh;
            delete[] requested_grid.yh;
            delete[] requested_grid.zh;

            delete[] ubuf;
            delete[] vbuf;
            delete[] wbuf;

            delete[] u_time_chunk;
            delete[] v_time_chunk;
            delete[] w_time_chunk;
        }

        else {
            MPI_Status status;
            MPI_Recv(parcels.xpos, parcels.nParcels*nTotTimes, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(parcels.ypos, parcels.nParcels*nTotTimes, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(parcels.zpos, parcels.nParcels*nTotTimes, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &status);

            delete[] requested_grid.xf;
            delete[] requested_grid.yf;
            delete[] requested_grid.zf;

            delete[] requested_grid.xh;
            delete[] requested_grid.yh;
            delete[] requested_grid.zh;

            delete[] ubuf;
            delete[] vbuf;
            delete[] wbuf;
        }

    }

    MPI_Finalize();
}
