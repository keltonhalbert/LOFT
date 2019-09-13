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
    float *pcluturb;
    float *pclvturb;
    float *pclwturb;
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
    float *pclxvortbaro;
    float *pclyvortbaro;
    float *pclzvortsolenoid;
    int nParcels;
    int nTimes;
};


/* This struct is used to hold the 4D arrays
 * used by the GPU to integrate the parcels 
 * and calculate various quantities/forcing 
 * terms along their paths. It has a
 * corresponding allocator and deallocator,
 * and is only ever used by Rank 0. This 
 * means that there shouldn't be any CPU
 * only code for this one. */
struct integration_data {

    float *u_4d_chunk;
    float *v_4d_chunk;
    float *w_4d_chunk;
    float *pi_4d_chunk;
    float *pres_4d_chunk;
    float *th_4d_chunk;
    float *rho_4d_chunk;
    float *rhof_4d_chunk;
    float *khh_4d_chunk;
    float *kmh_4d_chunk;

    float *turbu_4d_chunk;
    float *turbv_4d_chunk;
    float *turbw_4d_chunk;

    float *tem1_4d_chunk;
    float *tem2_4d_chunk;
    float *tem3_4d_chunk;
    float *tem4_4d_chunk;
    float *tem5_4d_chunk;
    float *tem6_4d_chunk;

    float *xvort_4d_chunk;
    float *yvort_4d_chunk;
    float *zvort_4d_chunk;

    float *xvtilt_4d_chunk;
    float *yvtilt_4d_chunk;
    float *zvtilt_4d_chunk;

    float *xvstretch_4d_chunk;
    float *yvstretch_4d_chunk;
    float *zvstretch_4d_chunk;
    float *turbxvort_4d_chunk;
    float *turbyvort_4d_chunk;
    float *turbzvort_4d_chunk;
    float *xvbaro_4d_chunk;
    float *yvbaro_4d_chunk;
 
    float *zvort_solenoid_4d_chunk; 
};


datagrid* allocate_grid_managed( int X0, int X1, int Y0, int Y1, int Z0, int Z1 );
datagrid* allocate_grid_cpu( int X0, int X1, int Y0, int Y1, int Z0, int Z1 );
void deallocate_grid_managed(datagrid *grid);
void deallocate_grid_cpu(datagrid *grid);
parcel_pos* allocate_parcels_managed(int NX, int NY, int NZ, int nTotTimes);
parcel_pos* allocate_parcels_cpu(int NX, int NY, int NZ, int nTotTimes);
void deallocate_parcels_managed(parcel_pos *parcels);
void deallocate_parcels_cpu(parcel_pos *parcels);
integration_data* allocate_integration_managed(long bufsize);
void deallocate_integration_managed(integration_data *data);
#endif
