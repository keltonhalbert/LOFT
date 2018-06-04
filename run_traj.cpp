#include "readlofs.cpp"
#include "loadseeds.cpp"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

/* Load the grid metadata and get the gird based on requested bounds.
 * This only needs to be called once to load data into memory. Additionally,
 * subsequent calls will look for cached metadata in .cm1hdf5_* files.
 */
void loadMetadataAndGrid(string base_dir, datagrid *requested_grid) {

    // query the dataset structure
    lofs_get_dataset_structure(base_dir);
    // get the HDF metadata - return the first filename
    get_hdf_metadata(firstfilename,&nx,&ny,&nz,&nodex,&nodey);

    // for right now, set the grid bounds to the saved
    // bounds for testing purposes
    requested_grid->X0 = saved_X0; requested_grid->Y0 = saved_Y0;
    requested_grid->X1 = saved_X1; requested_grid->Y1 = saved_Y1;
    requested_grid->Z0 = 0; requested_grid->Z1 = nz-1;

    // request a grid subset based on 
    // the subset information provided to
    // out grid struct
    lofs_get_grid(requested_grid);
}

/* Read in the U, V, and W vector components from the disk, provided previously allocated memory buffers
 * and the time requested in the dataset. 
 */
void loadVectorsFromDisk(datagrid *requested_grid, float *ubuffer, float *vbuffer, float *wbuffer, double t0) {
    // request 3D field!
    lofs_read_3dvar(requested_grid, ubuffer, (char *)"u", t0);
    lofs_read_3dvar(requested_grid, vbuffer, (char *)"v", t0);
    lofs_read_3dvar(requested_grid, wbuffer, (char *)"w", t0);
}

int main() {
    string base_dir = "/u/sciteam/halbert/project_bagm/khalbert/30m-every-time-step/3D";
    double t0 = 3001.;
    int nT = 45;
    datagrid requested_grid;
    loadMetadataAndGrid(base_dir, &requested_grid);

    cout << endl << "GRID DIMENSIONS" << endl;
    cout << "NX = " << requested_grid.NX << " NY = " << requested_grid.NY << " NZ = " << requested_grid.NZ << endl;
    cout << "X0 = " << requested_grid.xh[requested_grid.X0] << " Y0 = " << requested_grid.yh[requested_grid.Y0] << endl;
    cout << "X1 = " << requested_grid.xh[requested_grid.X1] << " Y1 = " << requested_grid.yh[requested_grid.Y1] << endl;
    cout << "Z0 = " << requested_grid.zh[requested_grid.Z0] << " Z1 = " << requested_grid.zh[requested_grid.Z1] << endl;

    // allocate space for U, V, and W arrays
    long bufsize = (long) (requested_grid.NX+1) * (long) (requested_grid.NY+1) * (long) (requested_grid.NZ+1) * (long) (nT) * (long) sizeof(float);
    cout << "Allocating Memory: " << bufsize * 1.25e-7 << "mb per vector for " << nT << " time steps" << endl;
    float *ubuffer = new float[(size_t)bufsize];
    float *vbuffer = new float[(size_t)bufsize];
    float *wbuffer = new float[(size_t)bufsize];

    for (int i = 0; i < nT; ++i) {

        cout << "TIMESTEP " << i << endl;
        loadVectorsFromDisk(&requested_grid, ubuffer, vbuffer, wbuffer, t0);
    }

    delete[] ubuffer;
    delete[] vbuffer;
    delete[] wbuffer;
}
