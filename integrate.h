#include <iostream>
#include "datastructs.cpp"
#ifndef INTEGRATE_H
#define INTEGRATE_H


void _nearest_grid_idx(float *point, float *x_grd, float *y_grd, float *z_grd, int *idx_4D, int nX, int nY, int nZ);
void cudaIntegrateParcels(datagrid grid, parcel_pos parcels, float *u_time_chunk, float *v_time_chunk, float *w_time_chunk, int MX, int MY, int MZ, int nT, int totTime);
#endif
