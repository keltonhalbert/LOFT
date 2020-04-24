#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"
#include <map>

extern "C" {
#include <lofs-read.h>
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#include <lofs-limits.h>
#include <lofs-macros.h>
}

#include "../include/datastructs.h"
#include "../include/integrate.h"
#include "../io/readlofs.cpp"
#include "../io/writenc.cpp"

/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

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

void nearest_grid_idx(grid *gd, mesh *msh, float *point, int *idx_4D) {

	int near_i = -1;
	int near_j = -1;
	int near_k = -1;

    float pt_x = point[0];
    float pt_y = point[1];
    float pt_z = point[2];


	// loop over the X grid
	for ( int i = -1; i < gd->NX-1; i++ ) {
		// find the nearest grid point index at X
		if ( ( pt_x >= xf(i) ) && ( pt_x <= xf(i+1) ) ) { near_i = i; } 
	}

	// loop over the Y grid
	for ( int j = -1; j < gd->NY-1; j++ ) {
		// find the nearest grid point index in the Y
		if ( ( pt_y >= yf(j) ) && ( pt_y <= yf(j+1) ) ) { near_j = j; } 
	}

	// loop over the Z grid
	for ( int k = 0; k < gd->NZ; k++ ) {
		// find the nearest grid point index in the Y
		if ( ( pt_z >= zf(k) ) && ( pt_z <= zf(k+1) ) ) { near_k = k; } 
	}

	// if a nearest index was not found, set all indices to -1 to flag
	// that the point is not in the domain
	if ((near_i == -1) || (near_j == -1) || (near_k == -1)) {
		near_i = -1; near_j = -1; near_k = -1;
	}

	idx_4D[0] = near_i; idx_4D[1] = near_j; idx_4D[2] = near_k;
	return;
}

/* Read a user supplied config/namelist file
 * used for specifying details about the parcel
 * seeds, data location, and variables to write. */
map<string, string> readCfg(string filename) {
    map<string, string> usrCfg;
    ifstream cFile(filename);
    if (cFile.is_open()) {
        string line;
        while(getline(cFile, line)) {
            line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());
            if (line[0] == '#' || line.empty()) continue;
            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            usrCfg[name] = value;
        }
    }
    else {
        cerr << "Couldn't open namelist file for reading." << endl;
    }
    return usrCfg;
}

/* Parse the user configuration and fill the variables with the necessary values */
void parse_cfg(map<string, string> *usrCfg, iocfg *io, string *histpath, string *base, int *verbose, int *debug, double *time, int *nTimes, \
            int *direction, float *X0, float *Y0, float *Z0, int *NX, int *NY, int *NZ, float *DX, float *DY, float *DZ) {
    *histpath = ((*usrCfg)["histpath"]);
    *base = ((*usrCfg)["basename"]);
	*verbose = stoi((*usrCfg)["verbose"]);
	*debug = stoi((*usrCfg)["debug"]);
    *X0 = stof((*usrCfg)["x0"]);
    *Y0 = stof((*usrCfg)["y0"]);
    *Z0 = stof((*usrCfg)["z0"]);
    *NX = stoi((*usrCfg)["nx"]);
    *NY = stoi((*usrCfg)["ny"]);
    *NZ = stoi((*usrCfg)["nz"]);
    *DX = stof((*usrCfg)["dx"]);
    *DY = stof((*usrCfg)["dy"]);
    *DZ = stof((*usrCfg)["dz"]);
    *time = stod((*usrCfg)["start_time"]);
    *nTimes = stoi((*usrCfg)["ntimesteps"]);
    *direction = stoi((*usrCfg)["time_direction"]);

    // Determine from the namelist file which variables
    // we need to read in for calculations or writing
    // along parcel traces, and store those values. 
    io->output_pbar = stoi((*usrCfg)["output_pbar"]);
    io->output_qvbar = stoi((*usrCfg)["output_qvbar"]);
    io->output_rhobar = stoi((*usrCfg)["output_rhobar"]);
    io->output_thetabar = stoi((*usrCfg)["output_thetabar"]);
    io->output_thrhobar = stoi((*usrCfg)["output_thrhobar"]);

    io->output_ppert = stoi((*usrCfg)["output_ppert"]);
    io->output_qvpert = stoi((*usrCfg)["output_qvpert"]);
    io->output_rhopert = stoi((*usrCfg)["output_rhopert"]);
    io->output_thetapert = stoi((*usrCfg)["output_thetapert"]);
    io->output_thrhopert = stoi((*usrCfg)["output_thrhopert"]);

    io->output_qc = stoi((*usrCfg)["output_qc"]);
    io->output_qi = stoi((*usrCfg)["output_qi"]);
    io->output_qs = stoi((*usrCfg)["output_qs"]);
    io->output_qg = stoi((*usrCfg)["output_qg"]);

    io->output_kmh = stoi((*usrCfg)["output_kmh"]);

    io->output_xvort = stoi((*usrCfg)["output_xvort"]);
    io->output_yvort = stoi((*usrCfg)["output_yvort"]);
    io->output_zvort = stoi((*usrCfg)["output_zvort"]);

    io->output_vorticity_budget = stoi((*usrCfg)["output_vorticity_budget"]);
    io->output_momentum_budget = stoi((*usrCfg)["output_momentum_budget"]);
}


/* Load the grid metadata and request a domain subset based on the 
 * current parcel positioning for the current time step. The idea is that 
 * for the first chunk of times read in (from 0 to N MPI ranks for time)
 * only the subset of the domain that matters for that period of time. 
 * When the next chunk of time is read in, check and see where the parcels
 * are currently and request a subset that is relevent to those parcels.   
 */
int getMeshBounds(string base_dir, dir_meta *dm, hdf_meta *hm, grid *gd, parcel_pos *parcels, int verbose, int rank) {
    if (verbose) cout << "Retrieving HDF Metadata" << endl;

	// Create a temporary full grid that we will then subset. We will
	// only do this in CPU memory because this will get deleted
	mesh *temp_msh;	
	sounding *temp_snd;

	if (verbose) cout << "Allocating temporary mesh" << endl;
	temp_msh = allocate_mesh_cpu(hm, gd);
	temp_snd = allocate_sounding_cpu(gd->NZ);

	// request the full grid so that we can find the indices
	// of where our parcels are, and then request a smaller
	// subset from there.
	if (verbose) cout << "Calling LOFS on temporary grid" << endl;
	lofs_get_grid(dm, hm, gd, temp_msh, temp_snd);

	// find the min/max index bounds of 
	// our parcels
	float point[3];
	int idx_4D[4];
	int min_i = gd->NX+1;
	int min_j = gd->NY+1;
	int min_k = gd->NZ+1;
	int max_i = -1;
	int max_j = -1;
	int max_k = -1;
	long invalid_pcls = 0;
	if (verbose) cout << "Searching the parcel bounds" << endl;
	for (int pcl = 0; pcl < parcels->nParcels; ++pcl) {
		point[0] = parcels->xpos[PCL(0, pcl, parcels->nTimes)];
		point[1] = parcels->ypos[PCL(0, pcl, parcels->nTimes)];
		point[2] = parcels->zpos[PCL(0, pcl, parcels->nTimes)];
		// find the nearest grid point!
		if ((point[0] == NC_FILL_FLOAT) || (point[1] == NC_FILL_FLOAT) || (point[2] == NC_FILL_FLOAT)) {
			++invalid_pcls;
			continue;
		}
		nearest_grid_idx(gd, temp_msh, point, idx_4D);
		if ( (idx_4D[0] == -1) || (idx_4D[1] == -1) || (idx_4D[2] == -1) ) {
			cout << "INVALID POINT X " << point[0] << " Y " << point[1] << " Z " << point[2] << endl;
			cout << "Parcel X " << parcels->xpos[PCL(0, pcl, parcels->nTimes)];
			cout << " Parcel Y " << parcels->ypos[PCL(0, pcl, parcels->nTimes)];
			cout << " Parcel Z " << parcels->zpos[PCL(0, pcl, parcels->nTimes)] << endl;
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
	if (verbose) cout << "Finished searching parcel bounds" << endl;
	// clear the memory from the temp grid
	if (verbose) cout << "Deallocating temporary grid" << endl;
	deallocate_mesh_cpu(temp_msh);
	deallocate_sounding_cpu(temp_snd);
	// If all of our parcels have left the domain, then the
	// invalid parcel counter will tell us. This will prevent
	// us from carrying on unecessarily. 
	if (invalid_pcls == parcels->nParcels) {
		return 0;
	}

	// we want to add a buffer to our dimensions so that
	// the parcels don't accidentally move outside of our
	// requested data. If the buffer goes outside the 
	// saved dimensions, set it to the saved dimensions.
	// We also do this for our staggered grid calculations
	min_i = gd->saved_X0 + min_i - 10;
	max_i = gd->saved_X0 + max_i + 10;
	min_j = gd->saved_Y0 + min_j - 10;
	max_j = gd->saved_Y0 + max_j + 10;
	min_k = gd->saved_Z0 + min_k - 10;
	max_k = gd->saved_Z0 + max_k + 10;
	if (verbose) {
		cout << "Attempted Parcel Bounds In Grid" << endl;
		cout << "X0: " << min_i << " X1: " << max_i << endl;
		cout << "Y0: " << min_j << " Y1: " << max_j << endl;
		cout << "Z0: " << min_k << " Z1: " << max_k << endl;
	}

	// keep the data in our saved bounds. 
	if (min_i < gd->saved_X0) min_i = gd->saved_X0;
	if (max_i > gd->saved_X1) max_i = gd->saved_X1;
	if (min_j < gd->saved_Y0) min_j = gd->saved_Y0;
	if (max_j > gd->saved_Y1) max_j = gd->saved_Y1;
	if (min_k < gd->saved_Z0) min_k = gd->saved_Z0;
	if (max_k > gd->saved_Z1) max_k = gd->saved_Z1;

	if (verbose) {
		cout << "Parcel Bounds In Grid" << endl;
		cout << "X0: " << min_i << " X1: " << max_i << endl;
		cout << "Y0: " << min_j << " Y1: " << max_j << endl;
		cout << "Z0: " << min_k << " Z1: " << max_k << endl;
	}
	// We need to set our grid attributes to
	// the new, smaller domain around the parcels.
	gd->X0 = min_i; gd->X1 = max_i;
	gd->Y0 = min_j; gd->Y1 = max_j;
	gd->Z0 = min_k; gd->Z1 = max_k;
	// We need to reset NX, NY, and NZ now
	gd->NX = gd->X1 - gd->X0 + 1; 
	gd->NY = gd->Y1 - gd->Y0 + 1;
	gd->NZ = gd->Z1 - gd->Z0 + 1;
	return 1;
}

/* Read in the U, V, and W vector components plus the buoyancy and turbulence fields 
 * from the disk, provided previously allocated memory buffers
 * and the time requested in the dataset. 
 */
void loadDataFromDisk(iocfg *io, dir_meta *dm, hdf_meta *hm, cmdline *cmd, grid *gd, \
						float *ustag, float *vstag, float *wstag, \
                        float *pbuffer, float *tbuffer, float *thbuffer, float *rhobuffer, \
                        float *qvbuffer, float *qcbuffer, float *qibuffer, float *qsbuffer, \
                        float *qgbuffer, float*kmhbuffer) {
	requested_cube rc;

	/* We select extra data for doing spatial averaging */
	/* By shrinking in saved_x0,saved_x1 etc by 1 point on either side (see set_span), this will not fail
	 * if we do not specify X0, X1 etc.*/

	rc.X0=gd->X0; rc.Y0=gd->Y0; rc.Z0=gd->Z0;
	rc.X1=gd->X1; rc.Y1=gd->Y1; rc.Z1=gd->Z1;

	rc.NX=rc.X1-rc.X0+1; rc.NY=rc.Y1-rc.Y0+1; rc.NZ=rc.Z1-rc.Z0;

	read_lofs_buffer(ustag,(char *)"u",*dm,*hm,rc,*cmd);
	read_lofs_buffer(vstag,(char *)"v",*dm,*hm,rc,*cmd);
	read_lofs_buffer(wstag,(char *)"w",*dm,*hm,rc,*cmd);

    if (io->output_momentum_budget || io->output_vorticity_budget || io->output_kmh ) {
		read_lofs_buffer(kmhbuffer,(char *)"kmh",*dm,*hm,rc,*cmd);
    }

    if (io->output_momentum_budget || io->output_vorticity_budget || io->output_ppert ) {
		read_lofs_buffer(pbuffer,(char *)"prespert",*dm,*hm,rc,*cmd);
    }
    if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thetapert ) {
		read_lofs_buffer(tbuffer,(char *)"thpert",*dm,*hm,rc,*cmd);
    }
    if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thrhopert ) {
		read_lofs_buffer(thbuffer,(char *)"thrhopert",*dm,*hm,rc,*cmd);
    }
    if (io->output_momentum_budget || io->output_vorticity_budget || io->output_rhopert ) {
		read_lofs_buffer(rhobuffer,(char *)"rhopert",*dm,*hm,rc,*cmd);
    }
    if (io->output_momentum_budget || io->output_vorticity_budget || io->output_qvpert ) {
		read_lofs_buffer(qvbuffer,(char *)"qvpert",*dm,*hm,rc,*cmd);
    }
    if (io->output_qc) read_lofs_buffer(qcbuffer,(char *)"qc",*dm,*hm,rc,*cmd);
    if (io->output_qs) read_lofs_buffer(qsbuffer,(char *)"qs",*dm,*hm,rc,*cmd);
    if (io->output_qi) read_lofs_buffer(qibuffer,(char *)"qi",*dm,*hm,rc,*cmd);
    if (io->output_qg) read_lofs_buffer(qgbuffer,(char *)"qg",*dm,*hm,rc,*cmd); 

}

/* Seed some parcels into the domain
 * in physical gridpoint space, and then
 * fill the remainder of the parcel traces
 * with missing values. 
 */
void seed_parcels(parcel_pos *parcels, float X0, float Y0, float Z0, int NX, int NY, int NZ, \
                    float DX, float DY, float DZ, int nTotTimes) {
    int nParcels = NX*NY*NZ;

    int pid = 0;
    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                parcels->xpos[PCL(0, pid, parcels->nTimes)] = X0 + i*DX;
                parcels->ypos[PCL(0, pid, parcels->nTimes)] = Y0 + j*DY;
                parcels->zpos[PCL(0, pid, parcels->nTimes)] = Z0 + k*DZ;
                pid += 1;
            }
        }
    }

    // fill the remaining portions of the array
    // with the missing value flag for the future
    // times that we haven't integrated to yet.
    for (int p = 0; p < nParcels; ++p) {
        for (int t = 1; t < parcels->nTimes; ++t) {
            parcels->xpos[PCL(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->ypos[PCL(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
            parcels->zpos[PCL(t, p, parcels->nTimes)] = NC_FILL_FLOAT;
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
    // initialize a bunch of MPI stuff.
    // Rank tells you which process
    // you are and size tells y ou how
    // many processes there are total
    int rank, size, debug, verbose;
    long N_stag, N_scal, MX, MY, MZ;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Barrier(MPI_COMM_WORLD);

    // our parcel struct containing 
    // the position arrays
    iocfg *io = new iocfg();
    parcel_pos *parcels;
    // variables for parcel seed locations,
    // amount, and spacing
    float pX0, pDX, pY0, pDY, pZ0, pDZ;
    int pNX, pNY, pNZ;
	bool parcelsAreInDomain;
    // what is our integration start point in the simulation?
    double time;
    int nTimeSteps;
    // parcel integration direction; default is forward
    int direct = 1;
    // variables for specifying our
    // data path and our output data 
    // path
    string base;
    string histpath;
    
    // parse the namelist options into the appropriate variables
    map<string, string> usrCfg = readCfg("parcel.namelist");
    parse_cfg(&usrCfg, io, &histpath, &base, &verbose, &debug, &time, &nTimeSteps, &direct, \
              &pX0, &pY0, &pZ0, &pNX, &pNY, &pNZ, &pDX, &pDY, &pDZ );

    string base_dir = histpath;
    string outfilename = string(base) + ".nc";
	dir_meta *dm = new dir_meta();
	hdf_meta *hm = new hdf_meta();
	grid *gd = new grid(); 
	cmdline *cmd = new cmdline();
	ncstruct *nc = new ncstruct();
	mesh *req_msh;
	sounding *snd;
	readahead *rh = new readahead();
	init_structs(cmd,dm,gd,nc,rh);

	cmd->verbose = verbose;
	cmd->debug = debug;


    int nTimeChunks = (int) (nTimeSteps / size); 
    if (nTimeSteps % size > 0) nTimeChunks += 1;

    // the number of time steps we have is 
    // the number of MPI ranks there are
    // plus the very last integration end time
    int nTotTimes = size+1;
    
	// Query our dataset structure.
	// If this has been done before, it reads
	// the information from cache files in the 
	// runtime directory. If it hasn't been run,
	// this step can take fair amount of time.
	lofs_get_dataset_structure(base_dir, dm, hm, gd, cmd, nc, rh);

    // This is the main loop that does the data reading and eventually
    // calls the CUDA code to integrate forward.
    for (int tChunk = 0; tChunk < nTimeChunks; ++tChunk) {

        // if this is the first chunk of time, seed the
        // parcel start locations
        if (tChunk == 0) {
            if (verbose) cout << "SEEDING PARCELS" << endl;
            if (rank == 0) {
                // allocate parcels on both CPU and GPU
                parcels = allocate_parcels_managed(io, pNX, pNY, pNZ, nTotTimes);
            }
            else {
                // for all other ranks, only
                // allocate on CPU
                parcels = allocate_parcels_cpu(io, pNX, pNY, pNZ, nTotTimes);
            }
            
            seed_parcels(parcels, pX0, pY0, pZ0, pNX, pNY, pNZ, pDX, pDY, pDZ, nTotTimes);
			if (verbose) cout << "FINISHED SEEDING PARCELS" << endl;
            // we also initialize the output netcdf file here
            if (rank == 0) init_nc(outfilename, parcels);
        }

        // Read in the metadata and request a mesh subset 
        // that is dynamically based on where our parcels
        // are in the simulation. This is done for all MPI
        // ranks so that they can request different time
        // steps, but only Rank 0 will allocate the grid
        // arrays on both the CPU and GPU.
		if (verbose) cout << "REQUESTING METADATA & MESH" << endl;
        
		parcelsAreInDomain = getMeshBounds(base_dir, dm, hm, gd, parcels, verbose, rank);
		if (!parcelsAreInDomain) {
			cout << "All parcels have exited the domain, or were not in the domain to begin with. Exiting the program." << endl;
			return 0;
		}
		// on rank zero, allocate our grid on both the
		// CPU and GPU so that the GPU knows something
		// about our data for future integration.
		if (rank == 0) {
			req_msh = allocate_mesh_managed( hm, gd );
			snd = allocate_sounding_managed( gd->NZ );
		}
		// For the other MPI ranks, we only need to
		// allocate the grids on the CPU for copying
		// data to the MPI_Gather call
		else {
			req_msh = allocate_mesh_cpu( hm, gd ); 
			snd = allocate_sounding_cpu( gd->NZ );
		}

		lofs_get_grid(dm, hm, gd, req_msh, snd);
		if (verbose) cout << "END METADATA & MESH REQUEST" << endl;

        // The number of grid points requested...
		N_stag = (gd->NX)*(gd->NY)*(gd->NZ);
		N_scal = (gd->NX)*(gd->NY)*(gd->NZ);


		float *ubuf, *vbuf, *wbuf, *pbuf, *tbuf, *thbuf, *rhobuf, *qvbuf, *qcbuf, *qibuf, *qsbuf, *qgbuf, *kmhbuf;

		// allocate space for U, V, and W arrays
		ubuf = new float[N_stag];
		vbuf = new float[N_stag];
		wbuf = new float[N_stag];
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_kmh) kmhbuf = new float[N_stag];
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_ppert) pbuf = new float[N_scal];
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thetapert) tbuf = new float[N_scal];
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thrhopert) thbuf = new float[N_scal];
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_rhopert) rhobuf = new float[N_scal];
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_qvpert) qvbuf = new float[N_scal];
		if (io->output_qc) qcbuf = new float[N_scal];
		if (io->output_qi) qibuf = new float[N_scal];
		if (io->output_qs) qsbuf = new float[N_scal];
		if (io->output_qg) qgbuf = new float[N_scal];


		// declare the struct on all ranks, but only
		// allocate space for it on Rank 0
		model_data *data;
		if (rank == 0) {
			data = allocate_model_managed(io, N_stag*size);
		}
		else {
			data = new model_data();
		}


		// we need to find the index of the nearest time to the user requested
		// time. If the index isn't found, abort.
		int nearest_tidx = find_nearest_index(dm->alltimes, time, dm->ntottimes);
		if (nearest_tidx < 0) {
			cout << "Invalid time index: " << nearest_tidx << " for time " << time << ". Abort." << endl;
			return 0;
		}
		//double dt = fabs(alltimes[nearest_tidx + direct*(1+tChunk*size)] - alltimes[nearest_tidx + direct*(tChunk*size)]);
		double dt = fabs(dm->alltimes[1] - dm->alltimes[0]);
		req_msh->dt = dt;
		// set the time value for LOFS
		cmd->time = dm->alltimes[nearest_tidx + direct*(rank + tChunk*size)];
		printf("TIMESTEP %d/%d %d %f dt= %f\n", rank+1, size, rank + tChunk*size, dm->alltimes[nearest_tidx + direct*( rank + tChunk*size)], dt);
		// load u, v, and w into memory
		loadDataFromDisk(io, dm, hm, cmd, gd, ubuf, vbuf, wbuf, pbuf, tbuf, thbuf, \
						 rhobuf, qvbuf, qcbuf, qibuf, qsbuf, qgbuf, kmhbuf);

		// for MPI runs that load multiple time steps into memory,
		// communicate the data you've read into our 4D array
		
		int senderr_u, senderr_v, senderr_w, senderr_kmh;
		int senderr_p, senderr_t, senderr_th, senderr_rho;
		int senderr_qv, senderr_qc, senderr_qi, senderr_qs, senderr_qg;

		senderr_u = MPI_Gather(ubuf, N_stag, MPI_FLOAT, data->ustag, N_stag, MPI_FLOAT, 0, MPI_COMM_WORLD);
		senderr_v = MPI_Gather(vbuf, N_stag, MPI_FLOAT, data->vstag, N_stag, MPI_FLOAT, 0, MPI_COMM_WORLD);
		senderr_w = MPI_Gather(wbuf, N_stag, MPI_FLOAT, data->wstag, N_stag, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_kmh) {
			senderr_kmh = MPI_Gather(kmhbuf, N_stag, MPI_FLOAT, data->kmh, N_stag, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}

		// Use N_scalar here so that there aren't random zeroes throughout the middle of the array
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_ppert) {
			senderr_p = MPI_Gather(pbuf, N_scal, MPI_FLOAT, data->prespert, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thetapert) {
			senderr_t = MPI_Gather(tbuf, N_scal, MPI_FLOAT, data->thetapert, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thrhopert) {
			senderr_th = MPI_Gather(thbuf, N_scal, MPI_FLOAT, data->thrhopert, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_rhopert) {
			senderr_rho = MPI_Gather(rhobuf, N_scal, MPI_FLOAT, data->rhopert, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_qvpert) {
			senderr_qv = MPI_Gather(qvbuf, N_scal, MPI_FLOAT, data->qvpert, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
		if (io->output_qc) senderr_qc = MPI_Gather(qcbuf, N_scal, MPI_FLOAT, data->qc, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (io->output_qi) senderr_qi = MPI_Gather(qibuf, N_scal, MPI_FLOAT, data->qi, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (io->output_qs) senderr_qs = MPI_Gather(qsbuf, N_scal, MPI_FLOAT, data->qs, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (io->output_qg) senderr_qg = MPI_Gather(qgbuf, N_scal, MPI_FLOAT, data->qg, N_scal, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// clean up temporary buffers
		delete[] ubuf;
		delete[] vbuf;
		delete[] wbuf;
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_ppert) {
			delete[] pbuf;
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thetapert) {
			delete[] tbuf;
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thrhopert) {
			delete[] thbuf;
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_rhopert) {
			delete[] rhobuf;
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_kmh) {
			delete[] kmhbuf;
		}
		if (io->output_momentum_budget || io->output_vorticity_budget || io->output_qvpert) {
			delete[] qvbuf;
		}
		if (io->output_qc) delete[] qcbuf;
		if (io->output_qi) delete[] qibuf;
		if (io->output_qs) delete[] qsbuf;
		if (io->output_qg) delete[] qgbuf;

		if (rank == 0) {
			if (verbose) {
				cout << "MPI Gather Error U: " << senderr_u << endl;
				cout << "MPI Gather Error V: " << senderr_v << endl;
				cout << "MPI Gather Error W: " << senderr_w << endl;
				if (io->output_momentum_budget || io->output_vorticity_budget || io->output_ppert) {
					cout << "MPI Gather Error P: " << senderr_p << endl;
				}
				if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thetapert) {
					cout << "MPI Gather Error T: " << senderr_t << endl;
				}
				if (io->output_momentum_budget || io->output_vorticity_budget || io->output_thrhopert) {
					cout << "MPI Gather Error TH: " << senderr_th << endl;
				}
				if (io->output_momentum_budget || io->output_vorticity_budget || io->output_rhopert) {
					cout << "MPI Gather Error RHO: " << senderr_rho << endl;
				}
				if (io->output_momentum_budget || io->output_vorticity_budget || io->output_kmh) {
					cout << "MPI Gather Error KMH: " << senderr_kmh << endl;
				}
				if (io->output_momentum_budget || io->output_vorticity_budget || io->output_qvpert) {
					cout << "MPI Gather Error QV: " << senderr_qv << endl;
				}
				if (io->output_qc) cout << "MPI Gather Error QC: " << senderr_qc << endl;
				if (io->output_qi) cout << "MPI Gather Error QI: " << senderr_qi << endl;
				if (io->output_qs) cout << "MPI Gather Error QS: " << senderr_qs << endl;
				if (io->output_qs) cout << "MPI Gather Error QG: " << senderr_qg << endl;
			}

			int nParcels = parcels->nParcels;
			if (verbose) cout << "Beginning parcel integration! Heading over to the GPU to do GPU things..." << endl;

			cudaIntegrateParcels(gd, req_msh, snd, data, parcels, NC_FILL_FLOAT, size, nTotTimes, direct); 
			if (verbose) cout << "Finished integrating parcels!" << endl;
			// write out our information to disk
			if (verbose) cout << "Beginning to write to disk..." << endl;
			write_parcels(outfilename, parcels, tChunk);

			// Now that we've integrated forward and written to disk, before we can go again
			// we have to set the current end position of the parcel to the beginning for 
			// the next leg of integration. Do that, and then reset all the other values
			// to missing.
			if (verbose) cout << "Setting final parcel position to beginning of array for next integration cycle..." << endl;
			for (int pcl = 0; pcl < parcels->nParcels; ++pcl) {
				parcels->xpos[PCL(0, pcl, parcels->nTimes)] = parcels->xpos[PCL(parcels->nTimes-1, pcl, parcels->nTimes)];
				parcels->ypos[PCL(0, pcl, parcels->nTimes)] = parcels->ypos[PCL(parcels->nTimes-1, pcl, parcels->nTimes)];
				parcels->zpos[PCL(0, pcl, parcels->nTimes)] = parcels->zpos[PCL(parcels->nTimes-1, pcl, parcels->nTimes)];
			}
			if (verbose) cout << "Parcel position arrays reset." << endl;

            // memory management for root rank
			deallocate_mesh_managed(req_msh);
			deallocate_sounding_managed(snd);
			deallocate_model_managed(io, data);
        }

        // house keeping for the non-master
        // MPI ranks
        else {
            // memory management
			deallocate_mesh_cpu(req_msh);
			deallocate_sounding_managed(snd);
        }

		gd->X0 = gd->saved_X0; gd->Y0 = gd->saved_Y0; gd->Z0 = gd->saved_Z0;
		gd->X1 = gd->saved_X1; gd->Y1 = gd->saved_Y1; gd->Z1 = gd->saved_Z1;
		gd->NX = gd->X1 - gd->X0 + 1;
		gd->NY = gd->Y1 - gd->Y0 + 1;
		gd->NZ = gd->Z1 - gd->Z0 + 1;
        // receive the updated parcel arrays
        // so that we can do proper subseting. This happens
        // after integration is complete from CUDA.
		MPI_Status status;
		MPI_Bcast(parcels->xpos, parcels->nParcels*nTotTimes, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(parcels->ypos, parcels->nParcels*nTotTimes, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(parcels->zpos, parcels->nParcels*nTotTimes, MPI_FLOAT, 0, MPI_COMM_WORLD);

    }

    if (rank == 0) {
		deallocate_parcels_managed(io, parcels);
        cout << "Finished!" << endl << endl;
    }
	else {
		deallocate_parcels_cpu(io, parcels);
	}
    MPI_Finalize();
}
