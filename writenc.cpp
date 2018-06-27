#ifndef WRITENC_CPP
#define WRITENC_CPP

#include "datastructs.cpp"
#include <iostream>
#include <string>
#include <netcdf>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

void init_nc(string filename, parcel_pos *parcels) {
    // Create the file.
    NcFile output(filename, NcFile::replace);

    NcDim pclDim = output.addDim("nParcels", parcels->nParcels);
    NcDim timeDim = output.addDim("nTimes");
    //NcDim recDim = test.addDim(REC_NAME);

    // define the coordinate variables
    vector<NcDim> gridDimVector;
    gridDimVector.push_back(pclDim);
    gridDimVector.push_back(timeDim);
    NcVar xVar = output.addVar("parcel_x_pos", ncFloat, gridDimVector);
    NcVar yVar = output.addVar("parcel_y_pos", ncFloat, gridDimVector);
    NcVar zVar = output.addVar("parcel_z_pos", ncFloat, gridDimVector);

    // Define the units attributes for coordinate vars. This
    // attatches a test attribute to each of the coordinate 
    // cariables containing the units
    xVar.putAtt("units", "meters from simulation origin");
    yVar.putAtt("units", "meters from simulation origin");
    zVar.putAtt("units", "meters from simulation origin");
}
 
void write_parcels(string filename, parcel_pos *parcels ) { 
    NcFile output(filename, NcFile::write);

    NcVar xVar = output.getVar("parcel_x_pos");
    NcVar yVar = output.getVar("parcel_y_pos");
    NcVar zVar = output.getVar("parcel_z_pos");

    vector<size_t> startp,countp;
    startp.push_back(0);
    startp.push_back(0);
    countp.push_back(parcels->nParcels);
    countp.push_back(parcels->nTimes);

    // Write the coordinate variable data to the file
    xVar.putVar(startp,countp,parcels->xpos);
    yVar.putVar(startp,countp,parcels->ypos);
    zVar.putVar(startp,countp,parcels->zpos);
    cout << "*** SUCCESS writing file " << filename << "!" << endl;
    return;
}

#endif

