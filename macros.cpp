
#include <iostream>
#include <stdio.h>
using namespace std;
#ifndef MACROS
#define MACROS

// these macros help us stay consistent
// with how indexing happens in CM1 for
// the stretched, staggered grids
#define UH(ix) uh[ix+1]
#define UF(ix) uf[ix+1]
#define VH(iy) vh[iy+1]
#define VF(iy) vf[iy+1]
#define MH(iz) mh[iz+1]
#define MF(iz) mf[iz+1]

// OK being a bit clever here ... fun with macros. This will make the
// // code a lot easier to compare to native CM1 Fortran90 code that we are
// // copying anyway. I adopt TEM for his tem array, UA for ua etc.
//
#define BUF(x,y,z) buf0[P3(x,y,z,NX,NY)]
#define TEM(x,y,z) dum0[P3(x,y,z,NX+1,NY+1)]
#define UA(x,y,z) ustag[P3(x+1,y+1,z,NX+2,NY+2)]
#define VA(x,y,z) vstag[P3(x+1,y+1,z,NX+2,NY+2)]
#define WA(x,y,z) wstag[P3(x+1,y+1,z,NX+2,NY+2)]

#define P2(t,p,mt) (((p)*(mt))+(t))
// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
#define P4(x,y,z,t,mx,my,mz) (((t)*(mx)*(my)*(mz))+((z)*(mx)*(my))+((y)*(mx))+(x))
#endif
