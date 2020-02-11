#ifndef CONSTAMTS_H
#define CONSTANTS_H

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


#endif
