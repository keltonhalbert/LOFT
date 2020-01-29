// header file for reading CUDA compiled
// stuffs
#ifndef DATASTRUCTS_H
#define DATASTRUCTS_H

// This data structure stores the I/O
// settings for which variables to read/write
// based on the desired calculations and output
struct iocfg {

    int output_pbar = 0;
    int output_qvbar = 0;
    int output_rhobar = 0;
    int output_thetabar = 0;
    int output_thrhobar = 0;

    int output_ppert = 0;
    int output_qvpert = 0;
    int output_rhopert = 0;
    int output_thetapert = 0;
    int output_thrhopert = 0;

    int output_qc = 0;
    int output_qi = 0;
    int output_qs = 0;
    int output_qg = 0;

    int output_kmh = 0;

    int output_xvort = 0;
    int output_yvort = 0;
    int output_zvort = 0;

    int output_vorticity_budget = 0;
    int output_momentum_budget = 0;
};

// this struct helps manage all the different
// attributes a grid or grid subset may have,
// including the staggered arrays, number of 
// points in eachd dimension, and the indices
// of the subset from the larger grid
//
// I'm also going to store base state grid
// variables in here since they are also 
// time independent, much like the other
// grid variables
struct datagrid {

    // 1D grid arrays
    float *xf;
    float *xh;
    float *yf;
    float *yh;
    float *zf;
    float *zh;

    float *u0;
    float *v0;
    float *qv0;
    float *th0;
    float *rho0;
    float *p0;
    
    float *uh;
    float *uf;
    float *vh;
    float *vf;
    float *mh;
    float *mf;

    int isValid;

    // dimensions for
    // the arrays
    long NX;
    long NY;
    long NZ;

    float dx;
    float dy;
    float dz;
    float dt;

    // the subset points of the grid
    // that this grid is a part of
    long X0; long Y0;
    long X1; long Y1;
    long Z0; long Z1;

};

struct parcel_pos {
    float *xpos;
    float *ypos;
    float *zpos;
    float *pclu;
    float *pclv;
    float *pclw;
    float *pclkmh;
    float *pcluturb;
    float *pclvturb;
    float *pclwturb;
    float *pcludiff;
    float *pclvdiff;
    float *pclwdiff;
    float *pclxvort;
    float *pclyvort;
    float *pclzvort;
    float *pclxvorttilt;
    float *pclyvorttilt;
    float *pclzvorttilt;
    float *pclxvortstretch;
    float *pclyvortstretch;
    float *pclzvortstretch;
    float *pclxvortturb;
    float *pclyvortturb;
    float *pclzvortturb;
    float *pclxvortdiff;
    float *pclyvortdiff;
    float *pclzvortdiff;
    float *pclxvortsolenoid;
    float *pclyvortsolenoid;
    float *pclzvortsolenoid;


    float *pclppert;
    float *pclqvpert;
    float *pclrhopert;
    float *pclthetapert;
    float *pclthrhopert;

    float *pclpbar;
    float *pclqvbar;
    float *pclrhobar;
    float *pclthetabar;
    float *pclthrhobar;

    float *pclqc;
    float *pclqi;
    float *pclqs;
    float *pclqg;

    int nParcels;
    int nTimes;
    iocfg *io;
};


/* This struct is used to hold the 4D arrays
 * used by the GPU to integrate the parcels 
 * and calculate various quantities/forcing 
 * terms along their paths. It has a
 * corresponding allocator and deallocator,
 * and is only ever used by Rank 0. This 
 * means that there shouldn't be any CPU
 * only code for this one. */
struct model_data {

    float *ustag;
    float *vstag;
    float *wstag;
    float *pi;
    float *prespert;
    float *thetapert;
    float *thrhopert;
    float *rhopert;
    float *rhof;
    float *kmh;
    float *qvpert;
    float *qc;
    float *qi;
    float *qs;
    float *qg;

    float *turbu;
    float *turbv;
    float *turbw;
    float *diffu;
    float *diffv;
    float *diffw;

    float *tem1;
    float *tem2;
    float *tem3;
    float *tem4;
    float *tem5;
    float *tem6;

    float *xvort;
    float *yvort;
    float *zvort;

    float *xvtilt;
    float *yvtilt;
    float *zvtilt;

    float *xvstretch;
    float *yvstretch;
    float *zvstretch;
    float *turbxvort;
    float *turbyvort;
    float *turbzvort;
    float *diffxvort;
    float *diffyvort;
    float *diffzvort;
 
    float *xvort_solenoid; 
    float *yvort_solenoid; 
    float *zvort_solenoid; 
    iocfg *io;
};

// These functions should only be compiled if 
// we're actually using a GPU... otherwise
// only expose the CPU functions
datagrid* allocate_grid_managed( int X0, int X1, int Y0, int Y1, int Z0, int Z1 );
void deallocate_grid_managed(datagrid *grid);
parcel_pos* allocate_parcels_managed(iocfg *io, int NX, int NY, int NZ, int nTotTimes);
void deallocate_parcels_managed(iocfg* io, parcel_pos *parcels);
model_data* allocate_model_managed(iocfg* io, long bufsize);
void deallocate_model_managed(iocfg* io, model_data *data);

datagrid* allocate_grid_cpu( int X0, int X1, int Y0, int Y1, int Z0, int Z1 );
void deallocate_grid_cpu(datagrid *grid);
parcel_pos* allocate_parcels_cpu(iocfg *io, int NX, int NY, int NZ, int nTotTimes);
void deallocate_parcels_cpu(iocfg *io, parcel_pos *parcels);

#endif
