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

/* Utility function to find the nearest index to a
 * time value provided. It searches the array for the
 * closest value to the one provided and returns the index.
 * If the value isn't found, the index is returned as -1.
 */
int find_nearest_index(double *arr, double val, int N) {
    int nearest = -1;
    float mindist = 9999;
    for (int idx = 0; idx < N; ++idx) {
        float diff = fabs(val - arr[idx]);
        if (diff < mindist) {
            mindist = diff;
            nearest = idx;
        }
    }
    return nearest;
}

// this was stolen from LOFS/cm1tools-3.0 hdf2.c
// under the parce_cmdline_hdf2nc function. I could
// have just linked to it, but I want the LOFS dependency
// to only be I/O so that other backends can be used, hence
// copying it here.

/* The command line parser takes information from the user
 * on where to put the parcel initial seeds, how many parcels,
 * and the spacing between parcels. It also is where the user
 * specifies the integration start time and duration.
 */
void parse_cmdline(int argc, char **argv, \
	char *histpath, char *base, double *time, int *nTimes, int *direction, \
	float *X0, float *Y0, float *Z0, int *NX, int *NY, int *NZ, \
    float *DX, float *DY, float *DZ )
{
	int got_histpath,got_base,got_time,got_ntimes,got_X0,got_NX,got_Y0,got_NY,got_Z0,got_NZ,got_DX,got_DY,got_DZ;
    int optcount=0;
	enum { OPT_HISTPATH = 1000, OPT_BASE, OPT_TIME, OPT_NTIMES, OPT_X0, OPT_Y0, OPT_Z0, OPT_NX, OPT_NY, OPT_NZ, OPT_DX,
		OPT_DY, OPT_DZ, OPT_DEBUG, OPT_DIRECTION };
	// see https://stackoverflow.com/questions/23758570/c-getopt-long-only-without-alias
	static struct option long_options[] =
	{
		{"histpath", required_argument, 0, OPT_HISTPATH},
		{"base",     required_argument, 0, OPT_BASE},
		{"time",     required_argument, 0, OPT_TIME},
		{"ntimes",   required_argument, 0, OPT_NTIMES},
		{"x0",       required_argument, 0, OPT_X0},
		{"y0",       required_argument, 0, OPT_Y0},
		{"z0",       required_argument, 0, OPT_Z0},
		{"nx",       required_argument, 0, OPT_NX},
		{"ny",       required_argument, 0, OPT_NY},
		{"nz",       required_argument, 0, OPT_NZ},
		{"dx",       required_argument, 0, OPT_DX},
		{"dy",       required_argument, 0, OPT_DY},
		{"dz",       required_argument, 0, OPT_DZ},
		{"direction",optional_argument, 0, OPT_DIRECTION},
		{"debug",    optional_argument, 0, OPT_DEBUG},
		{0, 0, 0, 0}//sentinel, needed!
	};

	got_histpath=got_base=got_time=got_ntimes=got_X0=got_NX=got_DX=got_Y0=got_NY=got_DY=got_Z0=got_NZ=got_DZ=0;

	int bail = 0;

    // print the usage information for the
    // command line options
	if (argc == 1)
	{
		fprintf(stderr,
		"Usage: %s --histpath=[histpath] --base=[base] --x0=[X0] --y0=[Y0] --z0=[Z0] --nx=[NX] --ny=[NY] --nz=[NZ] --dx=[DX] --dy=[DY] --dz=[DZ] --direction=[-1|1] --time=[time] --ntimes[nTimes]\n\
        --direction=-1 specifies backward trajectory, forward is default\n",argv[0]);
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
			case OPT_NTIMES:
				*nTimes = atoi(optarg);
				got_ntimes=1;
				printf("number of steps to integrate ntimes = %i\n",*nTimes);
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
			case OPT_DIRECTION:
				*direction = atoi(optarg);
				optcount++;
                printf("integration direction = %i\n", *direction);
				break;
			case '?':
				fprintf(stderr,"Exiting: unknown command line option.\n");
				exit(0);
				break;
		}
	}

		if (got_histpath==0) { fprintf(stderr,"--histpath not specified; I need to know where your data resides!\n"); bail = 1; }
		if (got_base==0)   { fprintf(stderr,"--base not specified; I need to know what you want your saved data called!\n"); bail = 1; }
		if (got_time==0)   { fprintf(stderr,"--time not specified; I need to know when to start integration!\n"); bail = 1; }
		if (got_ntimes==0)   { fprintf(stderr,"--ntimes not specified; I need to know how long to integrate for!\n"); bail = 1; }

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

    // load the saved grid dimmensions into 
    // the temporary grid, then we will find
    // a smaller subset to load into memory
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
    }
    requested_grid->isValid = 1;
    // clear the memory from the temp grid
    delete[] temp_grid.xh;
    delete[] temp_grid.yh;
    delete[] temp_grid.zh;
    delete[] temp_grid.xf;
    delete[] temp_grid.yf;
    delete[] temp_grid.zf;
    //delete[] temp_grid.th0;
    //delete[] temp_grid.qv0;
    // if literally all of our parcels aren't
    // in the domain then something has gone
    // horribly wrong
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
void loadVectorsFromDisk(datagrid *requested_grid, float *ubuffer, float *vbuffer, float *wbuffer, float *pbuffer, float *thbuffer, double t0) {
    // request 3D field!
    // u,v, and w are on their
    // respective staggered grids
    lofs_read_3dvar(requested_grid, ubuffer, (char *)"u", t0);
    lofs_read_3dvar(requested_grid, vbuffer, (char *)"v", t0);
    lofs_read_3dvar(requested_grid, wbuffer, (char *)"w", t0);
    lofs_read_3dvar(requested_grid, pbuffer, (char *)"prespert", t0);
    lofs_read_3dvar(requested_grid, thbuffer, (char *)"thrhopert", t0);
}


/* Seed some parcels into the domain
 * in physical gridpoint space, and then
 * fill the remainder of the parcel traces
 * with missing values. 
 */
void seed_parcels(parcel_pos *parcels, float X0, float Y0, float Z0, int NX, int NY, int NZ, \
                    float DX, float DY, float DZ, int nTotTimes) {
    int nParcels = NX*NY*NZ;

    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    parcels->xpos = new float[nParcels * nTotTimes];
    parcels->ypos = new float[nParcels * nTotTimes];
    parcels->zpos = new float[nParcels * nTotTimes];
    parcels->pclu = new float[nParcels * nTotTimes];
    parcels->pclv = new float[nParcels * nTotTimes];
    parcels->pclw = new float[nParcels * nTotTimes];
    parcels->pclppert = new float[nParcels * nTotTimes];
    parcels->pclthrhoprime = new float[nParcels * nTotTimes];
    parcels->pclxvort = new float[nParcels * nTotTimes];
    parcels->pclyvort = new float[nParcels * nTotTimes];
    parcels->pclzvort = new float[nParcels * nTotTimes];
    parcels->pclxvorttilt = new float[nParcels * nTotTimes];
    parcels->pclyvorttilt = new float[nParcels * nTotTimes];
    parcels->pclzvorttilt = new float[nParcels * nTotTimes];
    parcels->pclxvortstretch = new float[nParcels * nTotTimes];
    parcels->pclyvortstretch = new float[nParcels * nTotTimes];
    parcels->pclzvortstretch = new float[nParcels * nTotTimes];
    parcels->pclxvortbaro = new float[nParcels * nTotTimes];
    parcels->pclyvortbaro = new float[nParcels * nTotTimes];
    parcels->nParcels = nParcels;
    parcels->nTimes = nTotTimes;
    cout << X0 << " " << Y0 << " " << Z0 << endl;

    int pid = 0;
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                parcels->xpos[P2(0, pid, parcels->nTimes)] = X0 + i*DX;
                parcels->ypos[P2(0, pid, parcels->nTimes)] = Y0 + j*DY;
                parcels->zpos[P2(0, pid, parcels->nTimes)] = Z0 + k*DZ;
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
        }
    }
    cout << "END PARCEL SEED" << endl;

}

/* This parcel seed function is used
 * when trying to match parcel locations
 * to ones used in CM1. This is really
 * only used for testing purposes. */
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
    // variables for parcel seed locations,
    // amount, and spacing
    float pX0, pDX, pY0, pDY, pZ0, pDZ;
    int pNX, pNY, pNZ;
    // what is our integration start point in the simulation?
    double time;
    int nTimeSteps;
    // parcel integration direction; default is forward
    int direct = 1;
    // variables for specifying our
    // data path and our output data 
    // path
    char *base = new char[256];
    char *histpath = new char[256];
    
    // call the command line parser and store the user input in the
    // appropriate variables. Stole this and modified it from LOFS
    // because I'm too lazy to write my own and using a library I
    // have to link to sounds really cumbersome. 
    parse_cmdline(argc, argv, histpath, base, &time, &nTimeSteps, &direct, &pX0, &pY0, &pZ0, &pNX, &pNY, &pNZ, &pDX, &pDY, &pDZ );

    // convert the char arrays to strings
    // because this is C++ and we can use
    // objects in the 21st centuryc
    string base_dir = string(histpath);
    string outfilename = string(base) + ".nc";
    // get rid of useless dead weight
    // because we're good programmers
    delete[] base;
    delete[] histpath;

    int rank, size;
    long N, MX, MY, MZ;
    //int nTimeChunks = 120*2;

    // initialize a bunch of MPI stuff.
    // Rank tells you which process
    // you are and size tells y ou how
    // many processes there are total
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Barrier(MPI_COMM_WORLD);

    int nTimeChunks = (int) (nTimeSteps / size); // this is a temporary hack
    if (nTimeSteps % size > 0) nTimeChunks += 1;
    // to make the command line parser stuff work with the
    // existing code.
    // KELTON PLEASE REMEMBER TO CHANGE THIS SO THAT WE
    // DON'T HAVE 800000000 VARIABLES DOING THE SAME THING
    // LYING AROUND


    // the number of time steps we have is 
    // the number of MPI ranks there are
    // plus the very last integration end time
    int nTotTimes = size+1;
    // Query our dataset structure.
    // If this has been done before, it reads
    // the information from cache files in the 
    // runtime directory. If it hasn't been run,
    // this step can take fair amount of time.
    lofs_get_dataset_structure(base_dir);
    // our parcel struct containing 
    // the position arrays
    parcel_pos parcels;
    datagrid requested_grid;

    // This is the main loop that does the data reading and eventually
    // calls the CUDA code to integrate forward.
    for (int tChunk = 0; tChunk < nTimeChunks; ++tChunk) {
        // if this is the first chunk of time, seed the
        // parcel start locations
        if (tChunk == 0) {
            cout << "SEEDING PARCELS" << endl;
            //seed_parcels_cm1(&parcels, nTotTimes);
            
            // seed the parcel starting positions based on the command line
            // arguments provided by the user. I don't think any sanity checking is done
            // here for out of bounds positions so we probably need to be careful
            // of this and consider fixing that
            seed_parcels(&parcels, pX0, pY0, pZ0, pNX, pNY, pNZ, pDX, pDY, pDZ, nTotTimes);
            // we also initialize the output netcdf file here
            if (rank == 0) init_nc(outfilename, &parcels);
        }

        // read in the metadata and request a grid subset 
        // that is dynamically based on where our parcels
        // are in the simulation
        loadMetadataAndGrid(base_dir, &requested_grid, &parcels); 
        // read in some base state variables we will use
        if ((rank == 0) && (tChunk == 0)) {
            cout << requested_grid.qv0[0] << " " << requested_grid.th0[0] << endl;
        }
        if (requested_grid.isValid == 0) {
            cout << "Something went horribly wrong when requesting a domain subset. Abort." << endl;
            exit(-1);
        }


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
        float *ubuf, *vbuf, *wbuf, *pbuf, *thbuf;
        ubuf = new float[(size_t)bufsize];
        vbuf = new float[(size_t)bufsize];
        wbuf = new float[(size_t)bufsize];
        pbuf = new float[(size_t)bufsize];
        thbuf = new float[(size_t)bufsize];


        // construct a 4D contiguous array to store stuff in.
        // bufsize is the size of the 3D component and size is
        // the number of MPI ranks (which is also the number of times)
        // read in
        float *u_time_chunk, *v_time_chunk, *w_time_chunk, *p_time_chunk, *th_time_chunk; 
        if (rank == 0) {
            u_time_chunk = (float *) malloc ((size_t)bufsize*size);
            v_time_chunk = (float *) malloc ((size_t)bufsize*size);
            w_time_chunk = (float *) malloc ((size_t)bufsize*size);
            p_time_chunk = (float *) malloc ((size_t)bufsize*size);
            th_time_chunk = (float *) malloc ((size_t)bufsize*size);

        }

        // we need to find the index of the nearest time to the user requested
        // time. If the index isn't found, abort.
        int nearest_tidx = find_nearest_index(alltimes, time, ntottimes);
        if (nearest_tidx < 0) {
            cout << "Invalid time index: " << nearest_tidx << " for time " << time << ". Abort." << endl;
        }
        printf("TIMESTEP %d/%d %d %f\n", rank, size, rank + tChunk*size, alltimes[nearest_tidx + direct*( rank + tChunk*size)]);
        // load u, v, and w into memory
        loadVectorsFromDisk(&requested_grid, ubuf, vbuf, wbuf, pbuf, thbuf, alltimes[nearest_tidx + direct*(rank + tChunk*size)]);
        
        // for MPI runs that load multiple time steps into memory,
        // communicate the data you've read into our 4D array
        int senderr_u = MPI_Gather(ubuf, N, MPI_FLOAT, u_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_v = MPI_Gather(vbuf, N, MPI_FLOAT, v_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_w = MPI_Gather(wbuf, N, MPI_FLOAT, w_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_p = MPI_Gather(pbuf, N, MPI_FLOAT, p_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        int senderr_th = MPI_Gather(thbuf, N, MPI_FLOAT, th_time_chunk, N, MPI_FLOAT, 0, MPI_COMM_WORLD);


        if (rank == 0) {
            // send to the GPU!!
            int nParcels = parcels.nParcels;
            cudaIntegrateParcels(requested_grid, parcels, u_time_chunk, v_time_chunk, w_time_chunk, p_time_chunk, th_time_chunk, MX, MY, MZ, size, nTotTimes, direct); 
            // write out our information to disk
            write_parcels(outfilename, &parcels, tChunk);

            // Now that we've integrated forward and written to disk, before we can go again
            // we have to set the current end position of the parcel to the beginning for 
            // the next leg of integration. Do that, and then reset all the other values
            // to missing.
            for (int pcl = 0; pcl < parcels.nParcels; ++pcl) {
                parcels.xpos[P2(0, pcl, parcels.nTimes)] = parcels.xpos[P2(size, pcl, parcels.nTimes)];
                parcels.ypos[P2(0, pcl, parcels.nTimes)] = parcels.ypos[P2(size, pcl, parcels.nTimes)];
                parcels.zpos[P2(0, pcl, parcels.nTimes)] = parcels.zpos[P2(size, pcl, parcels.nTimes)];
                // empty out our parcel data arrays too
                parcels.pclu[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclv[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclw[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclppert[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclthrhoprime[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclxvort[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclyvort[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclzvort[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclxvorttilt[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclyvorttilt[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclzvorttilt[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclxvortstretch[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclyvortstretch[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclzvortstretch[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclxvortbaro[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
                parcels.pclyvortbaro[P2(0, pcl, parcels.nTimes)] = NC_FILL_FLOAT;
            }

            // memory management for root rank
            delete[] requested_grid.xf;
            delete[] requested_grid.yf;
            delete[] requested_grid.zf;

            delete[] requested_grid.xh;
            delete[] requested_grid.yh;
            delete[] requested_grid.zh;
            
            delete[] requested_grid.th0;
            delete[] requested_grid.qv0;

            delete[] ubuf;
            delete[] vbuf;
            delete[] wbuf;
            delete[] pbuf;
            delete[] thbuf;

            delete[] u_time_chunk;
            delete[] v_time_chunk;
            delete[] w_time_chunk;
            delete[] p_time_chunk;
            delete[] th_time_chunk;
        }

        // house keeping for the non-master
        // MPI ranks
        else {
            // memory management
            delete[] requested_grid.xf;
            delete[] requested_grid.yf;
            delete[] requested_grid.zf;

            delete[] requested_grid.xh;
            delete[] requested_grid.yh;
            delete[] requested_grid.zh;
            
            delete[] requested_grid.th0;
            delete[] requested_grid.qv0;

            delete[] ubuf;
            delete[] vbuf;
            delete[] wbuf;
            delete[] pbuf;
            delete[] thbuf;
        }
        // receive the updated parcel arrays
        // so that we can do proper subseting. This happens
        // after integration is complete from CUDA.
        MPI_Status status;
        MPI_Bcast(parcels.xpos, parcels.nParcels*nTotTimes, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(parcels.ypos, parcels.nParcels*nTotTimes, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(parcels.zpos, parcels.nParcels*nTotTimes, MPI_FLOAT, 0, MPI_COMM_WORLD);

    }

    MPI_Finalize();
}
