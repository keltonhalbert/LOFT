#include <iostream>
#include <fstream>
#include <string>

#include "mpi.h"
#include <stdio.h>
#include <string>

extern "C" {
#include <lofs-read.h>
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#include <lofs-limits.h>
#include <lofs-macros.h>
}

#include "../include/datastructs.h"
using namespace std;

#ifndef LOFS_READ
#define LOFS_READ
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

void lofs_get_dataset_structure(std::string base_dir, dir_meta *dm, hdf_meta *hm, grid *gd, cmdline *cmd, ncstruct *nc, readahead *rh) {

	int i,ni,nj,nk,status;
	char **argv;
	hid_t hdf_file_id;
	zfpacc zfpacc;

	/* begin */

	strcpy(cmd->histpath, base_dir.c_str());
	cmd->nvar_cmdline = 0;
	dm->regenerate_cache=0;

	if((realpath(cmd->histpath,dm->topdir))==NULL)
	{
		printf("%s: No such directory\n",cmd->histpath);
		exit(EXIT_FAILURE);
	}

	get_num_time_dirs(dm,*cmd); //Sets dm.ntimedirs

	/* Malloc our time directory arrays */
	dm->timedir = (char **)malloc(dm->ntimedirs * sizeof(char *));
	for (i=0; i < dm->ntimedirs; i++) {
		dm->timedir[i] = (char *)(malloc(MAXSTR * sizeof(char)));
	}
	dm->dirtimes = (double *)malloc(dm->ntimedirs * sizeof(double));

	get_sorted_time_dirs(dm,*cmd); //Sets dm.timedir char array
	get_num_node_dirs(dm,*cmd);    //Sets dm.nnodedirs

	dm->nodedir = (char **)malloc(dm->nnodedirs * sizeof(char *));
	for (i=0; i < dm->nnodedirs; i++) {
		dm->nodedir[i] = (char *)(malloc(8 * sizeof(char)));
	}
	// ORF 8 is 7 zero padded node number directory name plus 1 end of string char
	// TODO: make these constants/macros

	get_sorted_node_dirs(dm,*cmd); //Sets dm.nodedir char array

	get_saved_base(dm->timedir[0],dm->saved_base);

	//printf("Simulation identifier (saved_base) = %s\n",dm->saved_base);

	get_all_available_times(dm,gd,*cmd); //Gets all times in one double precision array

	if(cmd->debug)
	{
		printf("All available times: ");
		for (i=0; i<dm->ntottimes; i++)printf("%lf ",dm->alltimes[i]); printf("\n");
	}

	if ((hdf_file_id = H5Fopen (dm->firstfilename, H5F_ACC_RDONLY,H5P_DEFAULT)) < 0)
	{
		fprintf(stderr,"Unable to open %s, bailing!\n", dm->firstfilename);
		//ERROR_STOP("Can't open firstfilename! Weird...");
	} // Keep open as we need metadata, 1d sounding data, etc.

	get_hdf_metadata(*dm,hm,cmd,nc,argv,&hdf_file_id,&zfpacc);

	if (cmd->debug) {
		printf("nx = %i ny = %i nz = %i rankx = %i ranky = %i\n",hm->nx,hm->ny,hm->nz,hm->rankx,hm->ranky);
	}

	/* Check for idiocy and tweak the span (X0-X1/Y0-Y1/Z0-Z1) as necessary */
	set_span(gd,*hm,*cmd);
	gd->NX = gd->X1 - gd->X0 + 1;
	gd->NY = gd->Y1 - gd->Y0 + 1;
	gd->NZ = gd->Z1 - gd->Z0 + 1;
	//printf("X0: %d Y0: %d Z0: %d X1: %d Y1: %d Z1: %d\n", gd->X0, gd->Y0, gd->Z0, gd->X1, gd->Y1, gd->Z1);
	H5Fclose(hdf_file_id);
}


// get the grid info and return the volume subset of the
// grid we are interested in
void lofs_get_grid( dir_meta *dm, hdf_meta *hm, grid *gd, mesh *msh, sounding *snd ) {
	
	hid_t hdf_file_id;
	if ((hdf_file_id = H5Fopen (dm->firstfilename, H5F_ACC_RDONLY,H5P_DEFAULT)) < 0)
	{
		fprintf(stderr,"Unable to open %s, bailing!\n", dm->firstfilename);
		//ERROR_STOP("Can't open firstfilename! Weird...");
	} // Keep open as we need metadata, 1d sounding data, etc.

	// get the 1D arrays for the grid and base
	// state sounding from the HDF5 info
	set_1d_arrays(*hm,*gd,msh,snd,&hdf_file_id);
	H5Fclose(hdf_file_id);

}

void lofs_read_3dvar(grid *gd, mesh *msh, float *buffer, char *varname, bool istag, double t0) {
}

#endif


