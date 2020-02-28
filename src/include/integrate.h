#include <iostream>
#include "datastructs.h"
#ifndef INTEGRATE_H
#define INTEGRATE_H
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/


void _nearest_grid_idx(float *point, datagrid *grid, int *idx_4D);
void cudaIntegrateParcels(datagrid *grid, model_data *data, parcel_pos *parcels, int nT, int totTime, int direct);
#endif
