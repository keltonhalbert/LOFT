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
    cudaMallocManaged(&(grid->xf), (NX+2)*sizeof(float));
    cudaMallocManaged(&(grid->xh), (NX+2)*sizeof(float));

    cudaMallocManaged(&(grid->yf), (NY+2)*sizeof(float));
    cudaMallocManaged(&(grid->yh), (NY+2)*sizeof(float));

    // +2 is +1 for stagger, +1 for potential bottom ghost zone
    cudaMallocManaged(&(grid->zf), (NZ+2)*sizeof(float));
    cudaMallocManaged(&(grid->zh), (NZ+2)*sizeof(float));

    cudaMallocManaged(&(grid->uf), (NX+2)*sizeof(float));
    cudaMallocManaged(&(grid->uh), (NX+2)*sizeof(float));

    cudaMallocManaged(&(grid->vf), (NY+2)*sizeof(float));
    cudaMallocManaged(&(grid->vh), (NY+2)*sizeof(float));

    cudaMallocManaged(&(grid->mf), (NZ+2)*sizeof(float));
    cudaMallocManaged(&(grid->mh), (NZ+2)*sizeof(float));

    // allocate base state arrays
    cudaMallocManaged(&(grid->u0),   (NZ+1)*sizeof(float));
    cudaMallocManaged(&(grid->v0),   (NZ+1)*sizeof(float));
    cudaMallocManaged(&(grid->qv0),  (NZ+1)*sizeof(float));
    cudaMallocManaged(&(grid->th0),  (NZ+1)*sizeof(float));
    cudaMallocManaged(&(grid->rho0), (NZ+1)*sizeof(float));
    cudaMallocManaged(&(grid->p0),   (NZ+1)*sizeof(float));
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
    grid->xf = new float[NX+2];
    grid->xh = new float[NX+2];

    grid->yf = new float[NY+2];
    grid->yh = new float[NY+2];

    grid->zf = new float[NZ+2];
    grid->zh = new float[NZ+2];

    grid->uf = new float[NX+2];
    grid->uh = new float[NX+2];

    grid->vf = new float[NY+2];
    grid->vh = new float[NY+2];

    grid->mf = new float[NZ+2];
    grid->mh = new float[NZ+2];

    // allocate base state arrays
    grid->u0 = new float[NZ+1];
    grid->v0 = new float[NZ+1];
    grid->qv0 = new float[NZ+1];
    grid->th0 = new float[NZ+1];
    grid->rho0 = new float[NZ+1];
    grid->p0 = new float[NZ+1];

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
    cudaFree(grid->u0);
    cudaFree(grid->v0);
    cudaFree(grid->rho0);
    cudaFree(grid->th0);
    cudaFree(grid->qv0);
    cudaFree(grid->p0);
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
    delete[] grid->u0;
    delete[] grid->v0;
    delete[] grid->rho0;
    delete[] grid->th0;
    delete[] grid->qv0;
    delete[] grid->p0;
}

/* Allocate arrays for parcel info on both the CPU and GPU.
   This function should only be called by MPI Rank 0, so
   be sure to use the CPU function for Rank >= 1. */
parcel_pos* allocate_parcels_managed(iocfg *io, int NX, int NY, int NZ, int nTotTimes) {
    int nParcels = NX*NY*NZ;
    parcel_pos *parcels;
    // create the struct on both the GPU and the CPU.
    cudaMallocManaged(&parcels, sizeof(parcel_pos));
    cudaMallocManaged(&(parcels->io), sizeof(iocfg));
    // set the values of the struct on the GPU
    parcels->io->output_pbar = io->output_pbar; 
    parcels->io->output_qvbar = io->output_qvbar;
    parcels->io->output_rhobar = io->output_rhobar;
    parcels->io->output_thetabar = io->output_thetabar;
    parcels->io->output_thrhobar = io->output_thrhobar;

    parcels->io->output_ppert = io->output_ppert;
    parcels->io->output_qvpert = io->output_qvpert;
    parcels->io->output_rhopert = io->output_rhopert;
    parcels->io->output_thetapert = io->output_thetapert;
    parcels->io->output_thrhopert = io->output_thrhopert;

    parcels->io->output_qc = io->output_qc;
    parcels->io->output_qi = io->output_qi;
    parcels->io->output_qs = io->output_qs;
    parcels->io->output_qg = io->output_qg;

    parcels->io->output_xvort = io->output_xvort;
    parcels->io->output_yvort = io->output_yvort;
    parcels->io->output_zvort = io->output_zvort;

    parcels->io->output_kmh = io->output_kmh;

    parcels->io->output_vorticity_budget = io->output_vorticity_budget;
    parcels->io->output_momentum_budget = io->output_momentum_budget;
    
    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    cudaMallocManaged(&(parcels->xpos), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->ypos), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->zpos), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclu), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclv), nParcels*nTotTimes*sizeof(float)); 
    cudaMallocManaged(&(parcels->pclw), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_kmh ) cudaMallocManaged(&(parcels->pclkmh), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_momentum_budget) {
        cudaMallocManaged(&(parcels->pclupgrad), nParcels*nTotTimes*sizeof(float));
        cudaMallocManaged(&(parcels->pclvpgrad), nParcels*nTotTimes*sizeof(float));
        cudaMallocManaged(&(parcels->pclwpgrad), nParcels*nTotTimes*sizeof(float));
        cudaMallocManaged(&(parcels->pcluturb), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclvturb), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclwturb), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pcludiff), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclvdiff), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclwdiff), nParcels*nTotTimes*sizeof(float)); 
    }
    if (io->output_xvort || io->output_vorticity_budget) cudaMallocManaged(&(parcels->pclxvort), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_yvort || io->output_vorticity_budget) cudaMallocManaged(&(parcels->pclyvort), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_zvort || io->output_vorticity_budget) cudaMallocManaged(&(parcels->pclzvort), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_vorticity_budget) {
        cudaMallocManaged(&(parcels->pclxvorttilt), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclyvorttilt), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclzvorttilt), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclxvortstretch), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclyvortstretch), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclzvortstretch), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclxvortturb), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclyvortturb), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclzvortturb), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclxvortdiff), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclyvortdiff), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclzvortdiff), nParcels*nTotTimes*sizeof(float)); 
        cudaMallocManaged(&(parcels->pclxvortsolenoid), nParcels*nTotTimes*sizeof(float));
        cudaMallocManaged(&(parcels->pclyvortsolenoid), nParcels*nTotTimes*sizeof(float));
        cudaMallocManaged(&(parcels->pclzvortsolenoid), nParcels*nTotTimes*sizeof(float));
    }

    if (io->output_ppert) cudaMallocManaged(&(parcels->pclppert), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_qvpert) cudaMallocManaged(&(parcels->pclqvpert), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_rhopert) cudaMallocManaged(&(parcels->pclrhopert), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_thetapert) cudaMallocManaged(&(parcels->pclthetapert), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_thrhopert) cudaMallocManaged(&(parcels->pclthrhopert), nParcels*nTotTimes*sizeof(float)); 

    if (io->output_pbar) cudaMallocManaged(&(parcels->pclpbar), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_qvbar) cudaMallocManaged(&(parcels->pclqvbar), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_rhobar) cudaMallocManaged(&(parcels->pclrhobar), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_thetabar) cudaMallocManaged(&(parcels->pclthetabar), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_thrhobar) cudaMallocManaged(&(parcels->pclthrhobar), nParcels*nTotTimes*sizeof(float)); 

    if (io->output_qc) cudaMallocManaged(&(parcels->pclqc), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_qi) cudaMallocManaged(&(parcels->pclqi), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_qs) cudaMallocManaged(&(parcels->pclqs), nParcels*nTotTimes*sizeof(float)); 
    if (io->output_qg) cudaMallocManaged(&(parcels->pclqg), nParcels*nTotTimes*sizeof(float)); 

    // set the static variables
    parcels->nParcels = nParcels;
    parcels->nTimes = nTotTimes;
    cudaDeviceSynchronize();

    return parcels;
}

/* Allocate arrays only on the CPU for the grid. This is important
   for using with MPI, as only 1 rank should be allocating memory
   on the GPU */
parcel_pos* allocate_parcels_cpu(iocfg* io, int NX, int NY, int NZ, int nTotTimes) {
    int nParcels = NX*NY*NZ;
    parcel_pos *parcels = new parcel_pos();
    parcels->io = io;

    // allocate memory for the parcels
    // we are integrating for the entirety 
    // of the simulation.
    parcels->xpos = new float[nParcels*nTotTimes]; 
    parcels->ypos = new float[nParcels*nTotTimes]; 
    parcels->zpos = new float[nParcels*nTotTimes]; 
    parcels->pclu = new float[nParcels*nTotTimes]; 
    parcels->pclv = new float[nParcels*nTotTimes]; 
    parcels->pclw = new float[nParcels*nTotTimes]; 
    if (io->output_kmh) parcels->pclkmh = new float[nParcels*nTotTimes]; 
    if (io->output_momentum_budget) {
        parcels->pclupgrad = new float[nParcels*nTotTimes];
        parcels->pclvpgrad = new float[nParcels*nTotTimes];
        parcels->pclwpgrad = new float[nParcels*nTotTimes];
        parcels->pcluturb = new float[nParcels*nTotTimes]; 
        parcels->pclvturb = new float[nParcels*nTotTimes]; 
        parcels->pclwturb = new float[nParcels*nTotTimes]; 
        parcels->pcludiff = new float[nParcels*nTotTimes]; 
        parcels->pclvdiff = new float[nParcels*nTotTimes]; 
        parcels->pclwdiff = new float[nParcels*nTotTimes]; 
    }
    if (io->output_vorticity_budget || io->output_xvort) parcels->pclxvort = new float[nParcels*nTotTimes]; 
    if (io->output_vorticity_budget || io->output_yvort) parcels->pclyvort = new float[nParcels*nTotTimes]; 
    if (io->output_vorticity_budget || io->output_zvort) parcels->pclzvort = new float[nParcels*nTotTimes]; 
    if (io->output_vorticity_budget) {
        parcels->pclxvorttilt = new float[nParcels*nTotTimes]; 
        parcels->pclyvorttilt = new float[nParcels*nTotTimes]; 
        parcels->pclzvorttilt = new float[nParcels*nTotTimes]; 
        parcels->pclxvortstretch = new float[nParcels*nTotTimes]; 
        parcels->pclyvortstretch = new float[nParcels*nTotTimes]; 
        parcels->pclzvortstretch = new float[nParcels*nTotTimes]; 
        parcels->pclxvortturb = new float[nParcels*nTotTimes]; 
        parcels->pclyvortturb = new float[nParcels*nTotTimes]; 
        parcels->pclzvortturb = new float[nParcels*nTotTimes]; 
        parcels->pclxvortdiff = new float[nParcels*nTotTimes]; 
        parcels->pclyvortdiff = new float[nParcels*nTotTimes]; 
        parcels->pclzvortdiff = new float[nParcels*nTotTimes]; 
        parcels->pclxvortsolenoid = new float[nParcels*nTotTimes];
        parcels->pclyvortsolenoid = new float[nParcels*nTotTimes];
        parcels->pclzvortsolenoid = new float[nParcels*nTotTimes];
    }

    if (io->output_ppert) parcels->pclppert = new float[nParcels*nTotTimes];
    if (io->output_qvpert) parcels->pclqvpert = new float[nParcels*nTotTimes];
    if (io->output_rhopert) parcels->pclrhopert = new float[nParcels*nTotTimes];
    if (io->output_thetapert) parcels->pclthetapert = new float[nParcels*nTotTimes];
    if (io->output_thrhopert) parcels->pclthrhopert = new float[nParcels*nTotTimes];

    if (io->output_pbar) parcels->pclpbar = new float[nParcels*nTotTimes];
    if (io->output_qvbar) parcels->pclqvbar = new float[nParcels*nTotTimes];
    if (io->output_rhobar) parcels->pclrhobar = new float[nParcels*nTotTimes];
    if (io->output_thetabar) parcels->pclthetabar = new float[nParcels*nTotTimes];
    if (io->output_thrhobar) parcels->pclthrhobar = new float[nParcels*nTotTimes];

    if (io->output_qc) parcels->pclqc = new float[nParcels*nTotTimes];
    if (io->output_qi) parcels->pclqi = new float[nParcels*nTotTimes];
    if (io->output_qs) parcels->pclqs = new float[nParcels*nTotTimes];
    if (io->output_qg) parcels->pclqg = new float[nParcels*nTotTimes];
    // set the static variables
    parcels->nParcels = nParcels;
    parcels->nTimes = nTotTimes;

    return parcels;
}

/* Deallocate parcel arrays on both the CPU and the
   GPU */
void deallocate_parcels_managed(iocfg* io, parcel_pos *parcels) {
    cudaFree(parcels->xpos);
    cudaFree(parcels->ypos);
    cudaFree(parcels->zpos);
    cudaFree(parcels->pclu);
    cudaFree(parcels->pclv);
    cudaFree(parcels->pclw);
    if (io->output_kmh) cudaFree(parcels->pclkmh);
    if (io->output_momentum_budget) {
        cudaFree(parcels->pclupgrad);
        cudaFree(parcels->pclvpgrad);
        cudaFree(parcels->pclwpgrad);
        cudaFree(parcels->pcluturb);
        cudaFree(parcels->pclvturb);
        cudaFree(parcels->pclwturb);
        cudaFree(parcels->pcludiff);
        cudaFree(parcels->pclvdiff);
        cudaFree(parcels->pclwdiff);
    }
    if (io->output_vorticity_budget || io->output_xvort) cudaFree(parcels->pclxvort);
    if (io->output_vorticity_budget || io->output_yvort) cudaFree(parcels->pclyvort);
    if (io->output_vorticity_budget || io->output_zvort) cudaFree(parcels->pclzvort);
    if (io->output_vorticity_budget) {
        cudaFree(parcels->pclxvorttilt);
        cudaFree(parcels->pclyvorttilt);
        cudaFree(parcels->pclzvorttilt);
        cudaFree(parcels->pclxvortstretch);
        cudaFree(parcels->pclyvortstretch);
        cudaFree(parcels->pclzvortstretch);
        cudaFree(parcels->pclxvortturb);
        cudaFree(parcels->pclyvortturb);
        cudaFree(parcels->pclzvortturb);
        cudaFree(parcels->pclxvortdiff);
        cudaFree(parcels->pclyvortdiff);
        cudaFree(parcels->pclzvortdiff);
        cudaFree(parcels->pclxvortsolenoid);
        cudaFree(parcels->pclyvortsolenoid);
        cudaFree(parcels->pclzvortsolenoid);
    }

    if (io->output_ppert) cudaFree(parcels->pclppert);
    if (io->output_qvpert) cudaFree(parcels->pclqvpert);
    if (io->output_rhopert) cudaFree(parcels->pclrhopert);
    if (io->output_thetapert) cudaFree(parcels->pclthetapert);
    if (io->output_rhopert) cudaFree(parcels->pclthrhopert);

    if (io->output_pbar) cudaFree(parcels->pclpbar);
    if (io->output_qvbar) cudaFree(parcels->pclqvbar);
    if (io->output_rhobar) cudaFree(parcels->pclrhobar);
    if (io->output_thetabar) cudaFree(parcels->pclthetabar);
    if (io->output_thrhobar) cudaFree(parcels->pclthrhobar);

    if (io->output_qc) cudaFree(parcels->pclqc);
    if (io->output_qi) cudaFree(parcels->pclqi);
    if (io->output_qs) cudaFree(parcels->pclqs);
    if (io->output_qg) cudaFree(parcels->pclqg);

    cudaFree(parcels);
    cudaDeviceSynchronize();
}

/* Deallocate parcel arrays only on the CPU */
void deallocate_parcels_cpu(iocfg *io, parcel_pos *parcels) {
    delete[] parcels->xpos;
    delete[] parcels->ypos;
    delete[] parcels->zpos;
    delete[] parcels->pclu;
    delete[] parcels->pclv;
    delete[] parcels->pclw;
    if (io->output_kmh) delete[] parcels->pclkmh;
    if (io->output_momentum_budget) {
        delete[] parcels->pclupgrad;
        delete[] parcels->pclvpgrad;
        delete[] parcels->pclwpgrad;
        delete[] parcels->pcluturb;
        delete[] parcels->pclvturb;
        delete[] parcels->pclwturb;
        delete[] parcels->pcludiff;
        delete[] parcels->pclvdiff;
        delete[] parcels->pclwdiff;
    }
    if (io->output_vorticity_budget || io->output_xvort) delete[] parcels->pclxvort;
    if (io->output_vorticity_budget || io->output_yvort) delete[] parcels->pclyvort;
    if (io->output_vorticity_budget || io->output_zvort) delete[] parcels->pclzvort;
    if (io->output_vorticity_budget) {
        delete[] parcels->pclxvorttilt;
        delete[] parcels->pclyvorttilt;
        delete[] parcels->pclzvorttilt;
        delete[] parcels->pclxvortstretch;
        delete[] parcels->pclyvortstretch;
        delete[] parcels->pclzvortstretch;
        delete[] parcels->pclxvortturb;
        delete[] parcels->pclyvortturb;
        delete[] parcels->pclzvortturb;
        delete[] parcels->pclxvortdiff;
        delete[] parcels->pclyvortdiff;
        delete[] parcels->pclzvortdiff;
        delete[] parcels->pclxvortsolenoid;
        delete[] parcels->pclyvortsolenoid;
        delete[] parcels->pclzvortsolenoid;
    }

    if (io->output_ppert) delete[] parcels->pclppert;
    if (io->output_qvpert) delete[] parcels->pclqvpert;
    if (io->output_rhopert) delete[] parcels->pclrhopert;
    if (io->output_thetapert) delete[] parcels->pclthetapert;
    if (io->output_thrhopert) delete[] parcels->pclthrhopert;

    if (io->output_pbar) delete[] parcels->pclpbar;
    if (io->output_qvbar) delete[] parcels->pclqvbar;
    if (io->output_rhobar) delete[] parcels->pclrhobar;
    if (io->output_thetabar) delete[] parcels->pclthetabar;
    if (io->output_thrhobar) delete[] parcels->pclthrhobar;

    if (io->output_qc) delete[] parcels->pclqc;
    if (io->output_qi) delete[] parcels->pclqi;
    if (io->output_qs) delete[] parcels->pclqs;
    if (io->output_qg) delete[] parcels->pclqg;

    delete[] parcels;
}

/* Allocate the struct of 4D arrays that store
   fields for integration and calculation. This
   only ever gets called by Rank 0, so there 
   should be no need for a CPU counterpart. */
model_data* allocate_model_managed(iocfg *io, long bufsize) {
    model_data *data;
    // create the struct on both the GPU and the CPU.
    cudaMallocManaged(&data, sizeof(model_data));
    cudaMallocManaged(&(data->io), sizeof(iocfg));
    // set the values of the struct on the GPU
    data->io->output_pbar = io->output_pbar; 
    data->io->output_qvbar = io->output_qvbar;
    data->io->output_rhobar = io->output_rhobar;
    data->io->output_thetabar = io->output_thetabar;
    data->io->output_thrhobar = io->output_thrhobar;

    data->io->output_ppert = io->output_ppert;
    data->io->output_qvpert = io->output_qvpert;
    data->io->output_rhopert = io->output_rhopert;
    data->io->output_thetapert = io->output_thetapert;
    data->io->output_thrhopert = io->output_thrhopert;

    data->io->output_qc = io->output_qc;
    data->io->output_qi = io->output_qi;
    data->io->output_qs = io->output_qs;
    data->io->output_qg = io->output_qg;

    data->io->output_xvort = io->output_xvort;
    data->io->output_yvort = io->output_yvort;
    data->io->output_zvort = io->output_zvort;

    data->io->output_kmh = io->output_kmh;

    data->io->output_vorticity_budget = io->output_vorticity_budget;
    data->io->output_momentum_budget = io->output_momentum_budget;

    // Now, here we only allocate the arrays that we need based on the
    // user supplied namelist configuration. This should help with a)
    // not having to manually comment out the microphysics variables
    // every time, and b) save on memory load when possible. 

    // These are arrays that are 100% necessary for parcel integration.
    // The temporary arrays are included in this because pretty much any
    // secondary calculation requires at least one or more of these
    // arrays. So, better to just have them up front. 
    cudaMallocManaged(&(data->ustag), bufsize*sizeof(float));
    cudaMallocManaged(&(data->vstag), bufsize*sizeof(float));
    cudaMallocManaged(&(data->wstag), bufsize*sizeof(float));
    cudaMallocManaged(&(data->tem1), bufsize*sizeof(float));
    cudaMallocManaged(&(data->tem2), bufsize*sizeof(float));
    cudaMallocManaged(&(data->tem3), bufsize*sizeof(float));
    cudaMallocManaged(&(data->tem4), bufsize*sizeof(float));
    cudaMallocManaged(&(data->tem5), bufsize*sizeof(float));
    cudaMallocManaged(&(data->tem6), bufsize*sizeof(float));
    
    // Arrays that are optional depending on if they need to be tracked along
    // a parcel, or are part of a calculation/budget. 
    if (io->output_qc) cudaMallocManaged(&(data->qc), bufsize*sizeof(float));
    if (io->output_qi) cudaMallocManaged(&(data->qi), bufsize*sizeof(float));
    if (io->output_qs) cudaMallocManaged(&(data->qs), bufsize*sizeof(float));
    if (io->output_qg) cudaMallocManaged(&(data->qg), bufsize*sizeof(float));

    if (io->output_vorticity_budget || io->output_xvort) cudaMallocManaged(&(data->xvort), bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_yvort) cudaMallocManaged(&(data->yvort), bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_zvort) cudaMallocManaged(&(data->zvort), bufsize*sizeof(float));

    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_ppert) cudaMallocManaged(&(data->pi), bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_ppert) cudaMallocManaged(&(data->prespert), bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_thrhopert) cudaMallocManaged(&(data->thrhopert),  bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_thetapert) cudaMallocManaged(&(data->thetapert),  bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_rhopert) cudaMallocManaged(&(data->rhopert), bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_kmh) cudaMallocManaged(&(data->kmh), bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_qvpert) cudaMallocManaged(&(data->qvpert), bufsize*sizeof(float));
    if (io->output_vorticity_budget || io->output_momentum_budget) {
        cudaMallocManaged(&(data->rhof), bufsize*sizeof(float));
        cudaMallocManaged(&(data->pgradu), bufsize*sizeof(float));
        cudaMallocManaged(&(data->pgradv), bufsize*sizeof(float));
        cudaMallocManaged(&(data->pgradw), bufsize*sizeof(float));
        cudaMallocManaged(&(data->turbu), bufsize*sizeof(float));
        cudaMallocManaged(&(data->turbv), bufsize*sizeof(float));
        cudaMallocManaged(&(data->turbw), bufsize*sizeof(float));
        cudaMallocManaged(&(data->diffu), bufsize*sizeof(float));
        cudaMallocManaged(&(data->diffv), bufsize*sizeof(float));
        cudaMallocManaged(&(data->diffw), bufsize*sizeof(float));
    }
    if (io->output_vorticity_budget) {
        cudaMallocManaged(&(data->xvtilt), bufsize*sizeof(float));
        cudaMallocManaged(&(data->yvtilt), bufsize*sizeof(float));
        cudaMallocManaged(&(data->zvtilt), bufsize*sizeof(float));
        cudaMallocManaged(&(data->xvstretch), bufsize*sizeof(float));
        cudaMallocManaged(&(data->yvstretch), bufsize*sizeof(float));
        cudaMallocManaged(&(data->zvstretch), bufsize*sizeof(float));
        cudaMallocManaged(&(data->turbxvort), bufsize*sizeof(float));
        cudaMallocManaged(&(data->turbyvort), bufsize*sizeof(float));
        cudaMallocManaged(&(data->turbzvort), bufsize*sizeof(float));
        cudaMallocManaged(&(data->diffxvort), bufsize*sizeof(float));
        cudaMallocManaged(&(data->diffyvort), bufsize*sizeof(float));
        cudaMallocManaged(&(data->diffzvort), bufsize*sizeof(float));
        cudaMallocManaged(&(data->xvort_solenoid), bufsize*sizeof(float)); 
        cudaMallocManaged(&(data->yvort_solenoid), bufsize*sizeof(float)); 
        cudaMallocManaged(&(data->zvort_solenoid), bufsize*sizeof(float)); 
    }

    return data;

}

/* Deallocate the struct of 4D arrays that store
   fields for integration and calculation. This 
   only ever gets called by Rank 0, so there
   should be no need for a CPU counterpart. */
void deallocate_model_managed(iocfg *io, model_data *data) {
    cudaFree(data->ustag);
    cudaFree(data->vstag);
    cudaFree(data->wstag);
    cudaFree(data->tem1);
    cudaFree(data->tem2);
    cudaFree(data->tem3);
    cudaFree(data->tem4);
    cudaFree(data->tem5);
    cudaFree(data->tem6);

    if (io->output_qc) cudaFree(data->qc);
    if (io->output_qi) cudaFree(data->qi);
    if (io->output_qs) cudaFree(data->qs);
    if (io->output_qg) cudaFree(data->qg);

    if (io->output_vorticity_budget || io->output_xvort) cudaFree(data->xvort);
    if (io->output_vorticity_budget || io->output_yvort) cudaFree(data->yvort);
    if (io->output_vorticity_budget || io->output_zvort) cudaFree(data->zvort);

    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_ppert) cudaFree(data->pi);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_ppert) cudaFree(data->prespert);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_thrhopert) cudaFree(data->thrhopert);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_thetapert) cudaFree(data->thetapert);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_rhopert) cudaFree(data->rhopert);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_kmh) cudaFree(data->kmh);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_qvpert) cudaFree(data->qvpert);
    if (io->output_vorticity_budget || io->output_momentum_budget) {
        cudaFree(data->rhof);
        cudaFree(data->pgradu);
        cudaFree(data->pgradv);
        cudaFree(data->pgradw);
        cudaFree(data->turbu);
        cudaFree(data->turbv);
        cudaFree(data->turbw);
        cudaFree(data->diffu);
        cudaFree(data->diffv);
        cudaFree(data->diffw);
    }
    if (io->output_vorticity_budget) {
        cudaFree(data->xvtilt);
        cudaFree(data->yvtilt);
        cudaFree(data->zvtilt);
        cudaFree(data->xvstretch);
        cudaFree(data->yvstretch);
        cudaFree(data->zvstretch);
        cudaFree(data->turbxvort);
        cudaFree(data->turbyvort);
        cudaFree(data->turbzvort);
        cudaFree(data->diffxvort);
        cudaFree(data->diffyvort);
        cudaFree(data->diffzvort);
        cudaFree(data->xvort_solenoid); 
        cudaFree(data->yvort_solenoid); 
        cudaFree(data->zvort_solenoid); 
    }
}
#endif
