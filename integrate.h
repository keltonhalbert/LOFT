#include <iostream>
#include "datastructs.cpp"
#ifndef INTEGRATE_H
#define INTEGRATE_H
void cudaIntegrateParcels(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk,\
                            float *uparcels, float *vparcels, float *wparcels,  int MX, int MY, int MZ, int nT, int tChunk, int totTime);
#endif
