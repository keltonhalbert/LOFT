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

	hid_t hdf_file_id;

	/* begin */
	init_structs(&cmd,&dm,&gd,&nc,&rh);

	strcpy(cmd.histpath, base_dir.c_str());

	if((realpath(cmd.histpath,dm.topdir))==NULL)
	{
		printf("%s: No such directory\n",cmd.histpath);
		exit(EXIT_FAILURE);
	}

}


// get the grid info and return the volume subset of the
// grid we are interested in
void lofs_get_grid( datagrid *grid ) {
	
}

void lofs_read_3dvar(datagrid *grid, float *buffer, char *varname, bool istag, double t0) {
}

#endif


