#include "readlofs.cpp"
#include <iostream>
#include <string>

int main() {


    // hard coding our directory for testing purposes
    std::string base_dir = "/u/sciteam/halbert/project_bagm/khalbert/30m-kth/3D";

    // query the dataset structure
    lofs_get_dataset_structure(base_dir);
    // get the HDF metadata - return the first filename
    get_hdf_metadata(firstfilename,&nx,&ny,&nz,&nodex,&nodey);

    int t0 = (int)alltimes[0];

    // for right now, set the grid bounds to the saved
    // bounds for testing purposes
    datagrid requested_grid;
    requested_grid.X0 = saved_X0; requested_grid.Y0 = saved_Y0;
    requested_grid.X1 = saved_X1; requested_grid.Y1 = saved_Y1;
    requested_grid.Z0 = 0; requested_grid.Z1 = nz-1;

    // request a grid subset based on 
    // the subset information provided to
    // out grid struct
    lofs_get_grid(&requested_grid);


    // allocate space for U, V, and W arrays
    long bufsize = (long) (requested_grid.NX+1) * (long) (requested_grid.NY+1) * (long) (requested_grid.NZ+1) * (long) sizeof(float);
    float *ubuffer = new float[(size_t)bufsize];
    float *vbuffer = new float[(size_t)bufsize];
    float *wbuffer = new float[(size_t)bufsize];


    // request 3D field!
    lofs_read_3dvar(&requested_grid, ubuffer, (char *)"uinterp", t0);
    lofs_read_3dvar(&requested_grid, vbuffer, (char *)"vinterp", t0);
    lofs_read_3dvar(&requested_grid, wbuffer, (char *)"winterp", t0);

    // print some stuff to make sure it all worked properly
    std::cout << "GRID DIMS | NX = " << requested_grid.NX << " NY = " << requested_grid.NY << " NZ = " << requested_grid.NZ << std::endl;
    std::cout << "GRID SUBSET | X0 = " << requested_grid.X0 << " Y0 = " << requested_grid.Y0 << " Z0 = " << requested_grid.Z0 << std::endl;
    std::cout << "GRID SUBSET | X1 = " << requested_grid.X1 << " Y1 = " << requested_grid.Y1 << " Z1 = " << requested_grid.Z1 << std::endl;

    return 0;
}
