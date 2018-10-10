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

    NcVar uVar = output.addVar("pcl_u", ncFloat, gridDimVector);
    NcVar vVar = output.addVar("pcl_v", ncFloat, gridDimVector);
    NcVar wVar = output.addVar("pcl_w", ncFloat, gridDimVector);

    NcVar xvortVar = output.addVar("pcl_xvort", ncFloat, gridDimVector);
    NcVar yvortVar = output.addVar("pcl_yvort", ncFloat, gridDimVector);
    NcVar zvortVar = output.addVar("pcl_zvort", ncFloat, gridDimVector);

    // Define the units attributes for coordinate vars. This
    // attatches a test attribute to each of the coordinate 
    // cariables containing the units
    xVar.putAtt("units", "meters from simulation origin");
    yVar.putAtt("units", "meters from simulation origin");
    zVar.putAtt("units", "meters from simulation origin");

    uVar.putAtt("units", "meters / second");
    vVar.putAtt("units", "meters / second");
    wVar.putAtt("units", "meters / second");

    xvortVar.putAtt("units", "s^-1");
    yvortVar.putAtt("units", "s^-1");
    zvortVar.putAtt("units", "s^-1");
}
 
void write_parcels(string filename, parcel_pos *parcels, int writeIters ) { 
    NcFile output(filename, NcFile::write);

    NcVar xVar = output.getVar("parcel_x_pos");
    NcVar yVar = output.getVar("parcel_y_pos");
    NcVar zVar = output.getVar("parcel_z_pos");

    NcVar uVar = output.getVar("pcl_u");
    NcVar vVar = output.getVar("pcl_v");
    NcVar wVar = output.getVar("pcl_w");

    NcVar xvortVar = output.getVar("pcl_xvort");
    NcVar yvortVar = output.getVar("pcl_yvort");
    NcVar zvortVar = output.getVar("pcl_zvort");

    vector<size_t> startp,countp;
    startp.push_back(0);
    if (writeIters == 0) startp.push_back(0);
    else startp.push_back(parcels->nTimes * writeIters - writeIters);
    countp.push_back(parcels->nParcels);
    countp.push_back(parcels->nTimes);

    // Write the coordinate variable data to the file
    xVar.putVar(startp,countp,parcels->xpos);
    yVar.putVar(startp,countp,parcels->ypos);
    zVar.putVar(startp,countp,parcels->zpos);

    uVar.putVar(startp,countp,parcels->pclu);
    vVar.putVar(startp,countp,parcels->pclv);
    wVar.putVar(startp,countp,parcels->pclw);

    xvortVar.putVar(startp,countp,parcels->pclxvort);
    yvortVar.putVar(startp,countp,parcels->pclyvort);
    zvortVar.putVar(startp,countp,parcels->pclzvort);
    cout << "*** SUCCESS writing file " << filename << "!" << endl;
    return;
}

#endif

