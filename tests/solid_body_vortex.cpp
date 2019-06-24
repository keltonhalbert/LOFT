#ifndef SOLID_BODY
#define SOLID_BODY
#include <iostream>
#include <fstream>
#include <string>
#include "../macros.cpp"
#include "../datastructs.h"
using namespace std;
/* This code is used to test the trajectory calculations
 * for vorticity against some known field, in this case,
 * a solid body vortex. An artifical CM1 grid with no 
 * terrain is constructed and a steady-state flow field
 * imposed upon it. Trajectories are then seeded and moved
 * through the field, and then compared to an analytical 
 * solution based on the imposed field. */


/* Creates an artificial CM1 grid that has dimentions 
 * 10km x 10km x 2km (x, y, z) with an isotropic resolution
 * of 30 meters. Resolution is only because it's the same
 * as our actual dataset we will run this on. */
void create_grid(datagrid *grid) {
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float dx = grid->dx;
    float dy = grid->dy;
    float dz = grid->dz;

    // fill the scalar X coordinates
    for (int i = 0; i < NX; ++i) {
        grid->xh[i] = dx*i - 0.5*dx-0.5*dx*NX;
    }

    // fill the staggered X coordinates
    for (int i = 0; i < NX + 1; ++i) {
        grid->xf[i] = dx*(i-1) - 0.5*dx*NX;
    }

    // fill the scalar Y coordinates
    for (int j = 0; j < NY; ++j) {
        grid->yh[j] = dy*j - 0.5*dy-0.5*dy*NY;
    }

    // fill the staggered Y coordinates
    for (int j = 0; j < NY + 1; ++j) {
        grid->yf[j] = dy*(j-1) -0.5*dy*NY;
    }

    // fill the staggered Z coordinates
    for (int k = 0; k < NZ+1; ++k) {
        grid->zf[k] = dz*k;
        cout << "dz*k = " << dz*k << endl;
    }

    // fill the scalar Z coordinates
    for (int k = 0; k < NZ; ++k) {
        grid->zh[k] = 0.5*(grid->zf[k] + grid->zf[k+1]);
    }


    // fill the scale factor arrays
    for (int ix = 0; ix < grid->NX; ix++) grid->uh[ix] = dx/(grid->xf[ix+1]-grid->xf[ix]);
    for (int ix = 1; ix < grid->NX+1; ix++) grid->uf[ix] = dx/(grid->xh[ix]-grid->xh[ix-1]);
    for (int iy = 0; iy < grid->NY; iy++) grid->vh[iy] = dy/(grid->yf[iy+1]-grid->yf[iy]);
    for (int iy = 1; iy < grid->NY+1; iy++) grid->vf[iy] = dy/(grid->yh[iy]-grid->yh[iy-1]);
    grid->zf[0] = -grid->zf[2]; //param.F
    for (int iz = 0; iz <= grid->NZ; iz++) grid->mh[iz] = dz/(grid->zf[iz+1]-grid->zf[iz]);
    for (int iz = 1; iz <= grid->NZ+1; iz++) grid->mf[iz] = dz/(grid->zh[iz]-grid->zf[iz-1]);
    
}


/* This function runs the trajectory integrations
 * and compares them to the known result. */
int main(int argc, char **argv ) {
    float domain_extent = 10000.; // in meters or 20km 
    float domain_depth = 2000.; // in meters or 2km
    float dx = 30.; float dy = 30.; float dz = 30.;

    int NX = (int) (domain_extent / dx);
    int NY = (int) (domain_extent / dy);
    int NZ = (int) (domain_depth / dz);
    cout << "NX: " << NX << " NY: " << NY << " NZ: " << NZ << endl;


    datagrid *grid;
    grid = allocate_grid_managed( 0, NX, 0, NY, 0, NZ);
    grid->NX = NX;
    grid->NY = NY;
    grid->NZ = NZ;
    grid->dx = dx;
    grid->dy = dy;
    grid->dz = dz;

    create_grid(grid);
    cout << "XH: " << endl;
    for (int i = 0; i < NX; ++i) {
        cout << " " << grid->xh[i] << endl;
    }
    cout << endl;

    cout << "XF: " << endl;
    for (int i = 0; i < NX+1; ++i) {
        cout << " " << grid->xf[i] << endl;
    }
    cout << endl;

    cout << "YH: " << endl;
    for (int j = 0; j < NY; ++j) {
        cout << " " << grid->yh[j] << endl;
    }
    cout << endl;

    cout << "YF: " << endl;
    for (int j = 0; j < NY+1; ++j) {
        cout << " " << grid->yf[j] << endl;
    }
    cout << endl;

    cout << "ZH: " << endl;
    for (int k = 0; k < NZ; ++k) {
        cout << " " << grid->zh[k] << endl;
    }
    cout << endl;

    cout << "ZF: " << endl;
    for (int k = 0; k < NZ+1; ++k) {
        cout << " " << grid->zf[k] << endl;
    }
    cout << endl;

}

#endif
