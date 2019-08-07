#include <iostream>
#include <fstream>
#include <string>

#include "mpi.h"
#include <stdio.h>
#include <string>
#include "macros.cpp"
#include "datastructs.h"
extern "C" {
#include <lofs-read.h>
}

#ifndef LOFS_READ
#define LOFS_READ
// the following constants lifted from
// hdf2.c in LOFS
// TODO -- clean this up and make more
// C++ friendly because C is ugly
#define MAXVARIABLES (100)
#define MAXSTR (512)

char topdir[PATH_MAX+1];
int dn;
char **timedir; 
char **nodedir;
double *dirtimes;
int ntimedirs;
int nx,ny,nz,nodex,nodey;
char firstfilename[MAXSTR];
char base[MAXSTR];
int nnodedirs;
double *alltimes;
int ntottimes;
int firsttimedirindex;
int saved_X0,saved_Y0,saved_X1,saved_Y1;
const float MISSING=1.0E37;

int debug = 0;
int yes2d = 0;
int gzip = 0;
int filetype = NC_NETCDF4;
int saved_staggered_mesh_params = 1;
int nthreads = 1;


// just a simple 1D array print 
// used for some sanity checking
// of data reads
void print_arr(float *arr, int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// This is all lifted from the grok function in hdf2.c of LOFS
// but cleaned up a little bit to be more C++ esque with it's
// memory allocation because C++ is prettier
void lofs_get_dataset_structure(std::string base_dir) {

    // get the c string representation
    // of the directory for use with LOFS functions
    strcpy(topdir, base_dir.c_str());

    // query the number of directories corresponding to 
    // times in the dataset
    ntimedirs = get_num_time_dirs(topdir, debug, 0);
    cout << "MY TIME DIRS = " << ntimedirs << endl;

    // allocate an array containing
    // all of the directories for all times
    timedir = new char*[ntimedirs];
    for (int i = 0; i < ntimedirs; ++i) {
        timedir[i] = new char[MAXSTR];
    }
    
    // get the double representation of time times
    // from the filestructure
    dirtimes = new double[ntimedirs];
    get_sorted_time_dirs(topdir,timedir,dirtimes,ntimedirs,debug, 0);

    // query the number of directories corresponding to compute nodes
    nnodedirs =  get_num_node_dirs(topdir,timedir[0],debug, 0);
    // allocate and get the array of strings for the nodedirs
    nodedir = new char*[nnodedirs];
    // ORF 8 == 7 zero padded node number directory name plus 1 end of string char
    for (int i = 0; i < nnodedirs; ++i) {
        nodedir[i] = new char[8];
    }
    // sort the node directories
    get_sorted_node_dirs(topdir,timedir[0],nodedir,&dn,nnodedirs,debug, 0);

    // get all of the available times
    // from the dataset
    alltimes = get_all_available_times(topdir,timedir,ntimedirs,nodedir,nnodedirs,
                                        &ntottimes,firstfilename,&firsttimedirindex, 
                                        &saved_X0,&saved_Y0,&saved_X1,&saved_Y1,debug, 0);
}


// get the grid info and return the volume subset of the
// grid we are interested in
void lofs_get_grid( datagrid *grid ) {
	
	hid_t f_id;
	int NX,NY,NZ;
    int nk, nj, ni;
    // how many points are in our 
    // subset?
	NX = grid->NX;
	NY = grid->NY;
	NZ = grid->NZ;
    nk = NZ; nj = NY; ni = NX;

    // open the first found HDF5 files and use it to
    // construct our grid in memory. Since it's a self-describing
    // file system, only 1 file is needed to construct the whole
    // grid in order to then subset it. Yay for not having to
    // reconstruct the whole thing!!!
    f_id = H5Fopen(firstfilename, H5F_ACC_RDONLY,H5P_DEFAULT);

    // allocate memory for the grid arrays
    float *xhfull = new float[nx];
    float *yhfull = new float[ny];
    // staggered arrays have +1 to their
    // respective dimension
    float *xffull = new float[nx+1];
    float *yffull = new float[ny+1];

    float *zh = new float[nz+2];
    float *zf = new float[nz+2]; 
    float dx, dy, dz;
    float rdx, rdy, rdz;
    get0dfloat (f_id,(char *)"mesh/dx",&dx); rdx=1.0/dx;
    get0dfloat (f_id,(char *)"mesh/dy",&dy); rdy=1.0/dy;
    // temporary fix until this is saved by the model
    dz=dx;rdz=rdx;

    grid->dx = dx;
    grid->dy = dy;
    grid->dz = dz;

    // fill the arrays with the goods
    get1dfloat( f_id, (char *)"mesh/xhfull", xhfull, 0, nx );
    get1dfloat( f_id, (char *)"mesh/yhfull", yhfull, 0, ny );
    get1dfloat( f_id, (char *)"mesh/xffull", xffull, 0, nx+1 );
    get1dfloat( f_id, (char *)"mesh/yffull", yffull, 0, ny+1 );
    get1dfloat( f_id, (char *)"mesh/zh", zh, 0, nz );
    get1dfloat( f_id, (char *)"mesh/zf", zf, 0, nz+1 );
    float *qv0 = new float[nz];
    float *th0 = new float[nz];
    float *rho0 = new float[nz];
    get1dfloat(f_id, (char *)"basestate/qv0", qv0, 0, nz);
    get1dfloat(f_id, (char *)"basestate/th0", th0, 0, nz);
    get1dfloat(f_id, (char *)"basestate/rh0", rho0, 0, nz);




    // fill the z arrays with the subset portion
    // of the vertical dimension
	for (int iz = grid->Z0; iz <= grid->Z1; iz++) {
        grid->qv0[iz-grid->Z0] = qv0[iz];
        grid->th0[iz-grid->Z0] = th0[iz];
        grid->rho0[iz-grid->Z0] = rho0[iz];

    }


    // We recreate George's mesh/derivative calculation paradigm even though
    // // we are usually isotropic. We need to have our code here match what
    // // CM1 does internally for stretched and isotropic meshes.
    // //
    // // Becuase C cannot do have negative array indices (i.e., uh[-1]) like
    // // F90 can, we have to offset everything to keep the same CM1-like code
    // // We malloc enough space for the "ghost zones" and then make sure we
    // // offset by the correct amount on each side. The macros take care of
    // // the offsetting.

    // fill the x and y arrays with the subset
    // portion of the horizontal dimensions
    for (int ix = grid->X0 - 1; ix <= grid->X1 + 1; ix++) UH(ix-grid->X0) = dx/(xffull[ix+1]-xffull[ix]);
    for (int ix = grid->X0 - 1; ix <= grid->X1 + 1; ix++) UF(ix-grid->X0) = dx/(xhfull[ix]-xhfull[ix-1]);
    for (int iy = grid->Y0 - 1; iy <= grid->Y1 + 1; iy++) VH(iy-grid->Y0) = dy/(yffull[iy+1]-yffull[iy]);
    for (int iy = grid->Y0 - 1; iy <= grid->Y1 + 1; iy++) VF(iy-grid->Y0) = dy/(yhfull[iy]-yhfull[iy-1]);
    //zf[0] = -zf[2]; //param.F
    for (int iz = grid->Z0 - 1; iz <= grid->Z1 + 1; iz++) MH(iz-grid->Z0) = dz/(zf[iz+1]-zf[iz]);
    for (int iz = grid->Z0 - 1; iz <= grid->Z1 + 1; iz++) MF(iz-grid->Z0) = dz/(zh[iz]-zf[iz-1]);
    
    for (int iz = grid->Z0; iz <= grid->Z1; iz++) grid->zh[iz-grid->Z0] = zh[iz];
	for (int iy = grid->Y0; iy <= grid->Y1; iy++) grid->yh[iy-grid->Y0] = yhfull[iy];
	for (int ix = grid->X0; ix <= grid->X1; ix++) grid->xh[ix-grid->X0] = xhfull[ix];

    for (int iz = grid->Z0; iz <= grid->Z1+1; iz++) grid->zf[iz-grid->Z0] = zf[iz];
    for (int iy = grid->Y0; iy <= grid->Y1+1; iy++) grid->yf[iy-grid->Y0] = yffull[iy];
    for (int ix = grid->X0; ix <= grid->X1+1; ix++) grid->xf[ix-grid->X0] = xffull[ix];


    delete[] xffull;
    delete[] yffull;
    delete[] xhfull;
    delete[] yhfull;
    delete[] zh;
    delete[] zf;
    delete[] rho0;
    delete[] qv0;
    delete[] th0;
}

void lofs_read_3dvar(datagrid *grid, float *buffer, char *varname, bool istag, double t0) {

    // lifted from LOFS hdf2.c
    // topdir, timedir, nodedir, ntimedirs, dn, dirtimes, alltimes, ntottimes,
    // nodex, nodey all from lofs_get_dataset_structure
    //
    // X0, Y0, X1, Y1, Z0, Y1, nx, ny, nz all from lofs_get_grid
    if (istag) {
        //X0-1,Y0-1,X1+1,Y1+1,Z0,Z1,
        read_hdf_mult_md(buffer,topdir,timedir,nodedir,ntimedirs,dn,dirtimes,alltimes,ntottimes,t0,varname, \
                grid->X0-1,grid->Y0-1,grid->X1+1,grid->Y1+1,grid->Z0,grid->Z1+1,nx,ny,nz,nodex,nodey);
    }
    else {
        read_hdf_mult_md(buffer,topdir,timedir,nodedir,ntimedirs,dn,dirtimes,alltimes,ntottimes,t0,varname, \
                grid->X0,grid->Y0,grid->X1+1,grid->Y1+1,grid->Z0,grid->Z1+1,nx,ny,nz,nodex,nodey);
    }
}

#endif


