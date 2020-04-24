#include "datastructs.h"
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#ifndef PREFETCH_H
#define PREFETCH_H
void prefetch_parcels_cpu(iocfg *io, parcel_pos *parcels, cudaStream_t memStream);
void prefetch_model_gpu(iocfg *io, model_data *data, long bufsize, cudaStream_t memStream);
#endif
