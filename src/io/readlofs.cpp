#include <iostream>
#include <fstream>
#include <string>

#include "mpi.h"
#include <stdio.h>
#include <string>
#include "../include/macros.h"
#include "../include/datastructs.h"

extern "C" {
#include <lofs-read.h>
#include <dirstruct.h>
#include <hdf2nc.h>
#include <limits.h>
}
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

void lofs_get_dataset_structure(std::string base_dir) {

	int i,ni,nj,nk,status;

	dir_meta dm;
	hdf_meta hm;
	grid gd;
	cmdline cmd;
	ncstruct nc;
	mesh msh;
	sounding snd;
	readahead rh;
	buffers b;
	char **argv;

	hid_t hdf_file_id;

	/* begin */
	init_structs(&cmd,&dm,&gd,&nc,&rh);

	strcpy(cmd.histpath, base_dir.c_str());
	cmd.debug = true;
	cmd.nvar_cmdline = 0;
	dm.regenerate_cache=0;

	if((realpath(cmd.histpath,dm.topdir))==NULL)
	{
		printf("%s: No such directory\n",cmd.histpath);
		exit(EXIT_FAILURE);
	}

	get_num_time_dirs(&dm,cmd); //Sets dm.ntimedirs

	/* Malloc our time directory arrays */
	dm.timedir = (char **)malloc(dm.ntimedirs * sizeof(char *));
	for (i=0; i < dm.ntimedirs; i++)
		dm.timedir[i] = (char *)(malloc(MAXSTR * sizeof(char)));
	dm.dirtimes = (double *)malloc(dm.ntimedirs * sizeof(double));

	get_sorted_time_dirs(&dm,cmd); //Sets dm.timedir char array
	get_num_node_dirs(&dm,cmd);    //Sets dm.nnodedirs

	dm.nodedir = (char **)malloc(dm.nnodedirs * sizeof(char *));
	for (i=0; i < dm.nnodedirs; i++)
		dm.nodedir[i] = (char *)(malloc(8 * sizeof(char)));
	// ORF 8 is 7 zero padded node number directory name plus 1 end of string char
	// TODO: make these constants/macros

	get_sorted_node_dirs(&dm,cmd); //Sets dm.nodedir char array

	get_saved_base(dm.timedir[0],dm.saved_base);

	printf("Simulation identifier (saved_base) = %s\n",dm.saved_base);

	get_all_available_times(&dm,&gd,cmd); //Gets all times in one double precision array

	if(cmd.debug)
	{
		printf("All available times: ");
		for (i=0; i<dm.ntottimes; i++)printf("%lf ",dm.alltimes[i]); printf("\n");
	}

	if ((hdf_file_id = H5Fopen (dm.firstfilename, H5F_ACC_RDONLY,H5P_DEFAULT)) < 0)
	{
		fprintf(stderr,"Unable to open %s, bailing!\n", dm.firstfilename);
		//ERROR_STOP("Can't open firstfilename! Weird...");
	} // Keep open as we need metadata, 1d sounding data, etc.

	get_hdf_metadata(dm,&hm,&cmd,argv,&hdf_file_id);

	printf("3D variables available: ");
	for (i = 0; i < hm.nvar_available; i++)
		printf("%s ",hm.varname_available[i]);printf("\n");

	if(cmd.verbose&&cmd.nvar_cmdline > 0)
	{
		printf("We are requesting the following variables: ");
		for (i=0; i<cmd.nvar_cmdline; i++)
			printf("%s ",cmd.varname_cmdline[i]);
		 printf("\n");
	}

	if (cmd.debug) {
		printf("nx = %i ny = %i nz = %i nodex = %i nodey = %i\n",hm.nx,hm.ny,hm.nz,hm.nodex,hm.nodey);
	}

	/* Check for idiocy and tweak the span (X0-X1/Y0-Y1/Z0-Z1) as necessary */

	set_span(&gd,hm,cmd);
}


// get the grid info and return the volume subset of the
// grid we are interested in
void lofs_get_grid( datagrid *grid ) {
	
}

void lofs_read_3dvar(datagrid *grid, float *buffer, char *varname, bool istag, double t0) {
}

#endif


