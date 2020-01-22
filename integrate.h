#include <iostream>
#include "datastructs.h"
#ifndef INTEGRATE_H
#define INTEGRATE_H


void _nearest_grid_idx(float *point, datagrid *grid, int *idx_4D);
void cudaIntegrateParcels(datagrid *grid, model_data *data, parcel_pos *parcels, int nT, int totTime, int direct);
#endif
