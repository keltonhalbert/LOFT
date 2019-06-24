#ifndef SOLID_BODY
#define SOLID_BODY
#include <iostream>
#include <fstream>
#include <string>
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
    // fill the scalar X coordinates
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float dx = grid->dx;
    float dy = grid->dy;
    float dz = grid->dz;

    for (int i = 0; i < NX; ++i) {
        grid->xh[i] = dx*i - 0.5*dx-0.5*dx*NX;
    }

    // fill the staggered X coordinates
    for (int i = 0; i < NX + 1; ++i) {
        grid->xf[i] = dx*(i-1) - 0.5*dx*NX;
    }
    
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

    create_grid(grid);

}

#endif
