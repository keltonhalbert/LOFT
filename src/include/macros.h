#ifndef MACROS
#define MACROS

// these macros help us stay consistent
// with how indexing happens in CM1 for
// the stretched, staggered grids
#define UH(ix) grid->uh[ix+1]
#define UF(ix) grid->uf[ix+1]
#define VH(iy) grid->vh[iy+1]
#define VF(iy) grid->vf[iy+1]
#define MH(iz) grid->mh[iz]
#define MF(iz) grid->mf[iz]

#define xh(ix) grid->xh[ix+1]
#define xf(ix) grid->xf[ix+1]
#define yh(iy) grid->yh[iy+1]
#define yf(iy) grid->yf[iy+1]
#define zh(iz) grid->zh[iz]
#define zf(iz) grid->zf[iz]
// OK being a bit clever here ... fun with macros. This will make the
// // code a lot easier to compare to native CM1 Fortran90 code that we are
// // copying anyway. I adopt TEM for his tem array, UA for ua etc.
#define BUF4D(x,y,z,t) buf0[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]
#define TEM4D(x,y,z,t) dum0[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]
#define  UA4D(x,y,z,t) ustag[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]
#define  VA4D(x,y,z,t) vstag[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]
#define  WA4D(x,y,z,t) wstag[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]
#define  KM4D(x,y,z,t) kmstag[P4(x+1,y+1,z,t,NX+2,NY+2,NZ+1)]

#define BUF(x,y,z) buf0[P3(x+1,y+1,z,NX+2,NY+2)]
#define TEM(x,y,z) dum0[P3(x+1,y+1,z,NX+2,NY+2)]
#define  UA(x,y,z) ustag[P3(x+1,y+1,z,NX+2,NY+2)]
#define  VA(x,y,z) vstag[P3(x+1,y+1,z,NX+2,NY+2)]
#define  WA(x,y,z) wstag[P3(x+1,y+1,z,NX+2,NY+2)]
#define  KM(x,y,z) kmstag[P3(x+1,y+1,z,NX+2,NY+2)]

#define PCL(t,p,mt) (((p)*(mt))+(t))
// stole this define from LOFS
#define P3(x,y,z,mx,my) (((z)*(mx)*(my))+((y)*(mx))+(x))
// I made this myself by stealing from LOFS
#define P4(x, y, z, t, mx, my, mz) (((t)*(mx)*(my)*(mz))+((z)*(mx)*(my))+((y)*(mx))+(x))
#endif
