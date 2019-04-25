#include <vector>
#ifndef DATASTRUCTS
#define DATASTRUCTS
using namespace std;

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

    int isValid;

    // dimensions for
    // the arrays
    long NX;
    long NY;
    long NZ;

    // this is the full nz
    // used for the base state arrays
    long nz;

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

    float *pclxvort;
    float *pclyvort;
    float *pclzvort;
    float *pclxvorttilt;
    float *pclyvorttilt;
    float *pclzvorttilt;
    float *pclxvortstretch;
    float *pclyvortstretch;
    float *pclzvortstretch;
    float *pclxvortbaro;
    float *pclyvortbaro;

    float *pclppert;
    float *pclthrhoprime;

    int nParcels;
    int nTimes;
};
#endif
