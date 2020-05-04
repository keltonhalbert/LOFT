
extern "C" {
#include <lofs-read.h>
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#include <lofs-limits.h>
#include <lofs-macros.h>
}

#include "../include/datastructs.h"
#include "../include/prefetch.h"
using namespace std;

#ifndef PREFETCH 
#define PREFETCH 
void prefetch_parcels_cpu(iocfg *io, parcel_pos *parcels, cudaStream_t memStream) {
    int nParcels = parcels->nParcels;
    int nTotTimes = parcels->nTimes;
    cudaMemPrefetchAsync(parcels->xpos, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    cudaMemPrefetchAsync(parcels->ypos, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    cudaMemPrefetchAsync(parcels->zpos, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    cudaMemPrefetchAsync(parcels->pclu, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    cudaMemPrefetchAsync(parcels->pclv, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    cudaMemPrefetchAsync(parcels->pclw, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_kmh ) cudaMemPrefetchAsync(parcels->pclkmh, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_momentum_budget) {
        cudaMemPrefetchAsync(parcels->pclbuoy, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream);
        cudaMemPrefetchAsync(parcels->pclupgrad, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream);
        cudaMemPrefetchAsync(parcels->pclvpgrad, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream);
        cudaMemPrefetchAsync(parcels->pclwpgrad, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream);
        cudaMemPrefetchAsync(parcels->pcluturb, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclvturb, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclwturb, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pcludiff, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclvdiff, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclwdiff, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    }
    if (io->output_xvort || io->output_vorticity_budget) cudaMemPrefetchAsync(parcels->pclxvort, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_yvort || io->output_vorticity_budget) cudaMemPrefetchAsync(parcels->pclyvort, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_zvort || io->output_vorticity_budget) cudaMemPrefetchAsync(parcels->pclzvort, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_vorticity_budget) {
        cudaMemPrefetchAsync(parcels->pclxvorttilt, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclyvorttilt, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclzvorttilt, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclxvortstretch, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclyvortstretch, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclzvortstretch, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclxvortturb, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclyvortturb, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclzvortturb, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclxvortdiff, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclyvortdiff, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclzvortdiff, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclxvortbaro, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclyvortbaro, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
        cudaMemPrefetchAsync(parcels->pclxvortsolenoid, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream);
        cudaMemPrefetchAsync(parcels->pclyvortsolenoid, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream);
        cudaMemPrefetchAsync(parcels->pclzvortsolenoid, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream);
    }

    if (io->output_ppert) cudaMemPrefetchAsync(parcels->pclppert, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_qvpert) cudaMemPrefetchAsync(parcels->pclqvpert, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_rhopert) cudaMemPrefetchAsync(parcels->pclrhopert, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_thetapert) cudaMemPrefetchAsync(parcels->pclthetapert, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_thrhopert) cudaMemPrefetchAsync(parcels->pclthrhopert, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 

    if (io->output_pbar) cudaMemPrefetchAsync(parcels->pclpbar, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_qvbar) cudaMemPrefetchAsync(parcels->pclqvbar, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_rhobar) cudaMemPrefetchAsync(parcels->pclrhobar, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_thetabar) cudaMemPrefetchAsync(parcels->pclthetabar, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_thrhobar) cudaMemPrefetchAsync(parcels->pclthrhobar, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 

    if (io->output_qc) cudaMemPrefetchAsync(parcels->pclqc, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_qi) cudaMemPrefetchAsync(parcels->pclqi, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_qs) cudaMemPrefetchAsync(parcels->pclqs, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
    if (io->output_qg) cudaMemPrefetchAsync(parcels->pclqg, nParcels*nTotTimes*sizeof(float), cudaCpuDeviceId, memStream); 
}

void prefetch_model_gpu(iocfg *io, model_data *data, long bufsize, cudaStream_t memStream) {
	int device = -1;
	cudaGetDevice(&device);

    cudaMemPrefetchAsync(data->ustag, bufsize*sizeof(float), device, memStream);
    cudaMemPrefetchAsync(data->vstag, bufsize*sizeof(float), device, memStream);
    cudaMemPrefetchAsync(data->wstag, bufsize*sizeof(float), device, memStream);
    
    // Arrays that are optional depending on if they need to be tracked along
    // a parcel, or are part of a calculation/budget. 
    if (io->output_qc) cudaMemPrefetchAsync(data->qc, bufsize*sizeof(float), device, memStream);
    if (io->output_qi) cudaMemPrefetchAsync(data->qi, bufsize*sizeof(float), device, memStream);
    if (io->output_qs) cudaMemPrefetchAsync(data->qs, bufsize*sizeof(float), device, memStream);
    if (io->output_qg) cudaMemPrefetchAsync(data->qg, bufsize*sizeof(float), device, memStream);


    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_ppert) cudaMemPrefetchAsync(data->pipert, bufsize*sizeof(float), device, memStream);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_ppert) cudaMemPrefetchAsync(data->prespert, bufsize*sizeof(float), device, memStream);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_thrhopert) cudaMemPrefetchAsync(data->thrhopert,  bufsize*sizeof(float), device, memStream);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_thetapert) cudaMemPrefetchAsync(data->thetapert,  bufsize*sizeof(float), device, memStream);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_rhopert) cudaMemPrefetchAsync(data->rhopert, bufsize*sizeof(float), device, memStream);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_kmh) cudaMemPrefetchAsync(data->kmh, bufsize*sizeof(float), device, memStream);
    if (io->output_vorticity_budget || io->output_momentum_budget || io->output_qvpert) cudaMemPrefetchAsync(data->qvpert, bufsize*sizeof(float), device, memStream);
}
#endif
