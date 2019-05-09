// header file for reading CUDA compiled
// stuffs
#ifndef DATASTRUCTS_H
#define DATASTRUCTS_H
// this struct helps manage all the different
// attributes a grid or grib subset may have,
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

    float *qv0;
    float *th0;
    float *rho0;
    
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

    // the subset points of the grid
    // that this grid is a part of
    long X0; long Y0;
    long X1; long Y1;
    long Z0; long Z1;



};

struct parcel_pos {
    // create a vector of vectors. Each parcel is a
    // vector of floats for the position of the parcel,
    // and the container vector holds all parcels
    float *xpos;
    float *ypos;
    float *zpos;
    float *pclu;
    float *pclv;
    float *pclw;
    int nParcels;
    int nTimes;
};


datagrid* allocate_grid_managed( int X0, int X1, int Y0, int Y1, int Z0, int Z1 );
datagrid* allocate_grid_cpu( int X0, int X1, int Y0, int Y1, int Z0, int Z1 );
void deallocate_grid_managed(datagrid *grid);
void deallocate_grid_cpu(datagrid *grid);
parcel_pos* allocate_parcels_managed(int NX, int NY, int NZ, int nTotTimes);
parcel_pos* allocate_parcels_cpu(int NX, int NY, int NZ, int nTotTimes);
void deallocate_parcels_managed(parcel_pos *parcels);
void deallocate_parcels_cpu(parcel_pos *parcels);
#endif
