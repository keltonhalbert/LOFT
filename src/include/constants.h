#ifndef CONSTANTS_H
#define CONSTANTS_H

/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

/* These constants have been lifted from
 * CM1 r19.8 and are used in carious 
 * stencil and grid calculations.
 */

const float p00 = 1.0e5;
const float g = 9.81;
const float rd = 287.04;
const float cp = 1005.7;
const float cpv = 1870.0;
const float rv = 461.5;

const float rp00 = 1.0/p00;
const float cpinv = 1.0/cp;
const float rcp = 1.0/cp;
const float rovcp = rd/cp;
const float eps = rd/rv;
const float reps = rv/rd;

const float kdiff6 = 0.080;

#endif
