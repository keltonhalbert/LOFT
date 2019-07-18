
#include <iostream>
#include <stdio.h>
using namespace std;
#ifndef MACROS
#define MACROS

// these macros help us stay consistent
// with how indexing happens in CM1 for
// the stretched, staggered grids
#define UH(ix) grid->uh[ix+1]
#define UF(ix) grid->uf[ix+1]
#define VH(iy) grid->vh[iy+1]
#define VF(iy) grid->vf[iy+1]
#define MH(iz) grid->mh[iz+1]
#define MF(iz) grid->mf[iz+1]

// OK being a bit clever here ... fun with macros. This will make the
// // code a lot easier to compare to native CM1 Fortran90 code that we are
// // copying anyway. I adopt TEM for his tem array, UA for ua etc.
//
#define BUF4D(x,y,z,t) buf0[P4(x,y,z,t,NX,NY,NZ)]
#define TEM4D(x,y,z,t) dum0[P4(x,y,z,t,NX+1,NY+1,NZ+1)]
#define  UA4D(x,y,z,t) ustag[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]
#define  VA4D(x,y,z,t) vstag[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]
#define  WA4D(x,y,z,t) wstag[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]

#define PCL(t,p,mt) (((p)*(mt))+(t))
// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
#define P4(x, y, z, t, mx, my, mz) (((t)*(mx)*(my)*(mz))+((z)*(mx)*(my))+((y)*(mx))+(x))
#endif
