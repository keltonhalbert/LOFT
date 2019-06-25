#ifndef SOLID_BODY
#define SOLID_BODY
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
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

/* Create a staggered C-grid based on the provided number
 * of grid points in each dimension and spacing between them.
 * This code was primarily lifted from param.F in CM1. */
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
    }

    // fill the scalar Z coordinates
    for (int k = 0; k < NZ; ++k) {
        grid->zh[k] = 0.5*(grid->zf[k] + grid->zf[k+1]);
    }


    // fill the scale factor arrays
    for (int ix = 0; ix < NX; ix++) grid->uh[ix] = dx/(grid->xf[ix+1]-grid->xf[ix]);
    for (int iy = 0; iy < NY; iy++) grid->vh[iy] = dy/(grid->yf[iy+1]-grid->yf[iy]);
    for (int iz = 0; iz < NZ; iz++) grid->mh[iz] = dz/(grid->zf[iz+1]-grid->zf[iz]);
    for (int ix = 1; ix < NX; ix++) grid->uf[ix] = dx/(grid->xh[ix]-grid->xh[ix-1]);
    for (int iy = 1; iy < NY; iy++) grid->vf[iy] = dy/(grid->yh[iy]-grid->yh[iy-1]);
    grid->zf[0] = -grid->zf[2]; //param.F
    for (int iz = 1; iz < NZ; iz++) grid->mf[iz] = dz/(grid->zh[iz]-grid->zh[iz-1]);

    // a fix for the edges. This should be fine
    // as long as we're not testing stretched meshes here. 
    grid->uf[0] = grid->uf[1];
    grid->uf[NX] = grid->uf[NX-1];
    grid->vf[0] = grid->vf[1];
    grid->vf[NY] = grid->vf[NY-1];
    grid->mf[0] = grid->mf[1];
    grid->mf[NZ] = grid->mf[NZ-1];
    
}



/* Create a solid body body vortex on our grid. We do this
 * by creating a V-R solid body vortex in polar/cylindrical
 * coordinates and then convert the vectors into U/V cartesian
 * space. A solid body vortex has the same vorticity everywhere, 
 * so this should be a really simple test of the trajectory
 * calculations. */
void create_vortex(datagrid *grid, integration_data *data) {
    int NX = grid->NX;
    int NY = grid->NY;
    int NZ = grid->NZ;
    float omega = 0.01; // 1/s - our vertical vorticity value
    float r_xstag, r_ystag, theta, v_theta, v_r;

    // U staggered grid
    for (int i = 0; i < NX+1; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                r_xstag = sqrt(grid->xf[i]*grid->xf[i] + grid->yh[j]*grid->yh[j]);
                theta = atan(grid->yh[j] / grid->xf[i]);
                v_r = 0.0;
                v_theta = omega * r_xstag;
                // I'm leaving it in is most general form instead
                // of dropping the v_r term in case I decide to 
                // make use of this another way
                data->u_4d_chunk[P4(i, j, k, 0, NX, NY, NZ)] = v_r*cos(theta) - v_theta*sin(theta);
            }
        }
    }

    // V staggered grid
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY+1; ++j) {
            for (int k = 0; k < NZ; ++k) {
                r_ystag = sqrt(grid->xh[i]*grid->xh[i] + grid->yf[j]*grid->yf[j]);
                theta = atan(grid->yf[j] / grid->xh[i]);
                //cout << " r = " << r_ystag << " theta = " << theta << endl;
                v_r = 0.0;
                v_theta = omega * r_ystag;
                // I'm leaving it in is most general form instead
                // of dropping the v_r term in case I decide to 
                // make use of this another way
                data->v_4d_chunk[P4(i, j, k, 0, NX, NY, NZ)] = v_r*sin(theta) + v_theta*cos(theta);
            }
        }
    } 
}


/* This function runs the trajectory integrations
 * and compares them to the known result. */
int main(int argc, char **argv ) {
    /* Creates an artificial CM1 grid that has dimentions 
     * 10km x 10km x 2km (x, y, z) with an isotropic resolution
     * of 30 meters. Resolution is only because it's the same
     * as our actual dataset we will run this on. */
    float domain_extent = 10000.; // in meters or 20km 
    float domain_depth = 2000.; // in meters or 2km
    float dx = 30.; float dy = 30.; float dz = 30.;

    // get the number of grid points along each
    // dimension
    int NX = (int) (domain_extent / dx);
    int NY = (int) (domain_extent / dy);
    int NZ = (int) (domain_depth / dz);
    int N = NX*NY*NZ;
    cout << "NX: " << NX << " NY: " << NY << " NZ: " << NZ << endl;


    // allocate memory for our grid
    datagrid *grid;
    grid = allocate_grid_managed( 0, NX, 0, NY, 0, NZ);
    // set the grid attributes
    grid->NX = NX;
    grid->NY = NY;
    grid->NZ = NZ;
    grid->dx = dx;
    grid->dy = dy;
    grid->dz = dz;

    // fill the arrays with grid values
    create_grid(grid);

    // print them out for testing purposes
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

    cout << "UH: " << endl;
    for (int i = 0; i < NX; ++i) {
        cout << " " << grid->uh[i] << endl;
    }
    cout << endl;

    cout << "VH: " << endl;
    for (int j = 0; j < NY; ++j) {
        cout << " " << grid->vh[j] << endl;
    }
    cout << endl;

    cout << "MH: " << endl;
    for (int k = 0; k < NZ; ++k) {
        cout << " " << grid->mh[k] << endl;
    }
    cout << endl;

    cout << "UF: " << endl;
    for (int i = 0; i < NX+1; ++i) {
        cout << " " << grid->uf[i] << endl;
    }
    cout << endl;

    cout << "VF: " << endl;
    for (int j = 0; j < NY+1; ++j) {
        cout << " " << grid->vf[j] << endl;
    }
    cout << endl;

    cout << "MF: " << endl;
    for (int k = 0; k < NZ+1; ++k) {
        cout << " " << grid->mf[k] << endl;
    }
    cout << endl;

    integration_data *data;
    data = allocate_integration_managed(N);
    create_vortex(grid, data);
    for (int i = 0; i < NX+1; ++i) {
        for (int j = 0; j < NY; ++j) {
            cout << " " << data->u_4d_chunk[P4(i, j, 0, 0, NX, NY, NZ)] << " ";
        }
    }
    cout << endl;
}

#endif
