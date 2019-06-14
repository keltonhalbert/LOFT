#include <vector>
#include "macros.cpp"
#include "datastructs.h"
#include <iostream>
#ifndef DATASTRUCTS
#define DATASTRUCTS
using namespace std;

/* Allocate memory on the CPU and GPU for a grid. There are times,
    like for various MPI ranks, that you don't want to do this on both.
    See the similar function for doing this on just the CPU */
datagrid* allocate_grid_managed( int X0, int X1, int Y0, int Y1, int Z0, int Z1 ) {
    datagrid *grid;
    long NX, NY, NZ;

    cudaMallocManaged(&grid, sizeof(datagrid));
    grid->X0 = X0; grid->X1 = X1;
    grid->Y0 = Y0; grid->Y1 = Y1;
    grid->Z0 = Z0; grid->Z1 = Z1;

	NX = grid->X1 - grid->X0 + 1;
	NY = grid->Y1 - grid->Y0 + 1;
	NZ = grid->Z1 - grid->Z0 + 1;

    // set the grid attributes
    grid->NX = NX;
    grid->NY = NY;
    grid->NZ = NZ;

    // allocage grid arrays
    cudaMallocManaged(&(grid->xf), (NX+1)*sizeof(float));
    cudaMallocManaged(&(grid->xh), NX*sizeof(float));

    cudaMallocManaged(&(grid->yf), (NY+1)*sizeof(float));
    cudaMallocManaged(&(grid->yh), NY*sizeof(float));

    cudaMallocManaged(&(grid->zf), (NZ+1)*sizeof(float));
    cudaMallocManaged(&(grid->zh), NZ*sizeof(float));

    cudaMallocManaged(&(grid->uf), (NX+2)*sizeof(float));
    cudaMallocManaged(&(grid->uh), (NX+2)*sizeof(float));

    cudaMallocManaged(&(grid->vf), (NY+2)*sizeof(float));
    cudaMallocManaged(&(grid->vh), (NY+2)*sizeof(float));

    cudaMallocManaged(&(grid->mf), (NZ+2)*sizeof(float));
    cudaMallocManaged(&(grid->mh), (NZ+2)*sizeof(float));

    // allocate base state arrays
    cudaMallocManaged(&(grid->qv0), NZ*sizeof(float));
    cudaMallocManaged(&(grid->th0), NZ*sizeof(float));
    cudaMallocManaged(&(grid->rho0), NZ*sizeof(float));
    cudaDeviceSynchronize();

    return grid;
}

/* Allocate arrays only on the CPU for the grid. This is important
   for using with MPI, as only 1 rank should be allocating memory
   on the GPU */
datagrid* allocate_grid_cpu( int X0, int X1, int Y0, int Y1, int Z0, int Z1 ) {
    datagrid *grid = new datagrid();
    long NX, NY, NZ;

    grid->X0 = X0; grid->X1 = X1;
    grid->Y0 = Y0; grid->Y1 = Y1;
    grid->Z0 = Z0; grid->Z1 = Z1;

	NX = grid->X1 - grid->X0 + 1;
	NY = grid->Y1 - grid->Y0 + 1;
	NZ = grid->Z1 - grid->Z0 + 1;

    // set the grid attributes
    grid->NX = NX;
    grid->NY = NY;
    grid->NZ = NZ;

    // allocage grid arrays
    grid->xf = new float[NX+1];
    grid->xh = new float[NX];

    grid->yf = new float[NY+1];
    grid->yh = new float[NY];

    grid->zf = new float[NZ+1];
    grid->zh = new float[NZ];

    grid->uf = new float[NX+2];
    grid->uh = new float[NX+2];

    grid->vf = new float[NY+2];
    grid->vh = new float[NY+2];

    grid->mf = new float[NZ+2];
    grid->mh = new float[NZ+2];

    // allocate base state arrays
    grid->qv0 = new float[NZ];
    grid->th0 = new float[NZ];
    grid->rho0 = new float[NZ];

    return grid;
}

/* Deallocate all of the arrays in the 
   struct for both the GPU and CPU */
void deallocate_grid_managed(datagrid *grid) {
    cudaFree(grid->xf);
    cudaFree(grid->xh);
    cudaFree(grid->yf);
    cudaFree(grid->yh);
    cudaFree(grid->zf);
    cudaFree(grid->zh);
    cudaFree(grid->uf);
    cudaFree(grid->uh);
    cudaFree(grid->vf);
    cudaFree(grid->vh);
    cudaFree(grid->mf);
    cudaFree(grid->mh);
    cudaFree(grid->rho0);
    cudaFree(grid->th0);
    cudaFree(grid->qv0);
    cudaDeviceSynchronize();
}

/* Deallocate all of the arrays in the
   struct only for the CPU */
void deallocate_grid_cpu(datagrid *grid) {
    delete[] grid->xf;
    delete[] grid->xh;
    delete[] grid->yf;
    delete[] grid->yh;
    delete[] grid->zf;
    delete[] grid->zh;
    delete[] grid->uf;
    delete[] grid->uh;
    delete[] grid->vf;
    delete[] grid->vh;
    delete[] grid->mf;
    delete[] grid->mh;
    delete[] grid->rho0;
    delete[] grid->th0;
    delete[] grid->qv0;
}

/* Allocate arrays for parcel info on both the CPU and GPU.
   This function should only be called by MPI Rank 0, so
   be sure to use the CPU function for Rank >= 1. */
parcel_pos* allocate_parcels_managed(int NX, int NY, int NZ, int nTotTimes) {
    int nParcels = NX*NY*NZ;
    parcel_pos *parcels;
    // create the struct on both the GPU and the CPU.
    cudaMallocManaged(&parcels, sizeof(parcel_pos));

    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    cudaMallocManaged(&(parcels->xpos), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->ypos), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->zpos), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclu), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclv), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclw), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclxvort), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclyvort), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclzvort), nParcels*nTotTimes*sizeof(float)); 
    // set the static variables
    parcels->nParcels = nParcels;
    parcels->nTimes = nTotTimes;
    cudaDeviceSynchronize();

    return parcels;
}

/* Allocate arrays only on the CPU for the grid. This is important
   for using with MPI, as only 1 rank should be allocating memory
   on the GPU */
parcel_pos* allocate_parcels_cpu(int NX, int NY, int NZ, int nTotTimes) {
    int nParcels = NX*NY*NZ;
    parcel_pos *parcels = new parcel_pos();

    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    parcels->xpos = new float[nParcels*nTotTimes]; 
    parcels->ypos = new float[nParcels*nTotTimes]; 
    parcels->zpos = new float[nParcels*nTotTimes]; 
    parcels->pclu = new float[nParcels*nTotTimes]; 
    parcels->pclv = new float[nParcels*nTotTimes]; 
    parcels->pclw = new float[nParcels*nTotTimes]; 
    parcels->pclxvort = new float[nParcels*nTotTimes]; 
    parcels->pclyvort = new float[nParcels*nTotTimes]; 
    parcels->pclzvort = new float[nParcels*nTotTimes]; 
    // set the static variables
    parcels->nParcels = nParcels;
    parcels->nTimes = nTotTimes;

    return parcels;
}

/* Deallocate parcel arrays on both the CPU and the
   GPU */
void deallocate_parcels_managed(parcel_pos *parcels) {
    cudaFree(parcels->xpos);
    cudaFree(parcels->ypos);
    cudaFree(parcels->zpos);
    cudaFree(parcels->pclu);
    cudaFree(parcels->pclv);
    cudaFree(parcels->pclw);
    cudaFree(parcels->pclxvort);
    cudaFree(parcels->pclyvort);
    cudaFree(parcels->pclzvort);
    cudaFree(parcels);
    cudaDeviceSynchronize();
}

/* Deallocate parcel arrays only on the CPU */
void deallocate_parcels_cpu(parcel_pos *parcels) {
    delete[] parcels->xpos;
    delete[] parcels->ypos;
    delete[] parcels->zpos;
    delete[] parcels->pclu;
    delete[] parcels->pclv;
    delete[] parcels->pclw;
    delete[] parcels->pclxvort;
    delete[] parcels->pclyvort;
    delete[] parcels->pclzvort;
    delete[] parcels;
}

/* Allocate the struct of 4D arrays that store
   fields for integration and calculation. This
   only ever gets called by Rank 0, so there 
   should be no need for a CPU counterpart. */
integration_data* allocate_integration_managed(int bufsize) {
    integration_data *data;
    // create the struct on both the GPU and the CPU.
    cudaMallocManaged(&data, sizeof(integration_data));

    // allocate the arrays in the struct
    cudaMallocManaged(&(data->u_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->v_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->w_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->pres_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->th_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->rho_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->khh_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->xvort_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->yvort_4d_chunk), bufsize*sizeof(float));
    cudaMallocManaged(&(data->zvort_4d_chunk), bufsize*sizeof(float));
    cudaDeviceSynchronize();

    return data;

}

/* Deallocate the struct of 4D arrays that store
   fields for integration and calculation. This 
   only ever gets called by Rank 0, so there
   should be no need for a CPU counterpart. */
void deallocate_integration_managed(integration_data *data) {
    cudaFree(data->u_4d_chunk);
    cudaFree(data->v_4d_chunk);
    cudaFree(data->w_4d_chunk);
    cudaFree(data->pres_4d_chunk);
    cudaFree(data->th_4d_chunk);
    cudaFree(data->rho_4d_chunk);
    cudaFree(data->khh_4d_chunk);
    cudaFree(data->xvort_4d_chunk);
    cudaFree(data->yvort_4d_chunk);
    cudaFree(data->zvort_4d_chunk);
    cudaDeviceSynchronize();
}
#endif
