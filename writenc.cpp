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
    NcVar xVar = output.addVar("xpos", ncFloat, gridDimVector);
    NcVar yVar = output.addVar("ypos", ncFloat, gridDimVector);
    NcVar zVar = output.addVar("zpos", ncFloat, gridDimVector);

    NcVar uVar = output.addVar("u", ncFloat, gridDimVector);
    NcVar vVar = output.addVar("v", ncFloat, gridDimVector);
    NcVar wVar = output.addVar("w", ncFloat, gridDimVector);

    NcVar xvortVar = output.addVar("xvort", ncFloat, gridDimVector);
    NcVar yvortVar = output.addVar("yvort", ncFloat, gridDimVector);
    NcVar zvortVar = output.addVar("zvort", ncFloat, gridDimVector);
    NcVar xvorttiltVar = output.addVar("xvorttilt", ncFloat, gridDimVector);
    NcVar yvorttiltVar = output.addVar("yvorttilt", ncFloat, gridDimVector);
    NcVar zvorttiltVar = output.addVar("zvorttilt", ncFloat, gridDimVector);
    NcVar xvortstretchVar = output.addVar("xvortstretch", ncFloat, gridDimVector);
    NcVar yvortstretchVar = output.addVar("yvortstretch", ncFloat, gridDimVector);
    NcVar zvortstretchVar = output.addVar("zvortstretch", ncFloat, gridDimVector);

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
    xvorttiltVar.putAtt("units", "s^-2");
    yvorttiltVar.putAtt("units", "s^-2");
    zvorttiltVar.putAtt("units", "s^-2");
    xvortstretchVar.putAtt("units", "s^-2");
    yvortstretchVar.putAtt("units", "s^-2");
    zvortstretchVar.putAtt("units", "s^-2");
}
 
void write_parcels(string filename, parcel_pos *parcels, int writeIters ) { 
    NcFile output(filename, NcFile::write);

    NcVar xVar = output.getVar("xpos");
    NcVar yVar = output.getVar("ypos");
    NcVar zVar = output.getVar("zpos");

    NcVar uVar = output.getVar("u");
    NcVar vVar = output.getVar("v");
    NcVar wVar = output.getVar("w");

    NcVar xvortVar = output.getVar("xvort");
    NcVar yvortVar = output.getVar("yvort");
    NcVar zvortVar = output.getVar("zvort");
    NcVar xvorttiltVar = output.getVar("xvorttilt");
    NcVar yvorttiltVar = output.getVar("yvorttilt");
    NcVar zvorttiltVar = output.getVar("zvorttilt");
    NcVar xvortstretchVar = output.getVar("xvortstretch");
    NcVar yvortstretchVar = output.getVar("yvortstretch");
    NcVar zvortstretchVar = output.getVar("zvortstretch");

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
    xvorttiltVar.putVar(startp,countp,parcels->pclxvorttilt);
    yvorttiltVar.putVar(startp,countp,parcels->pclyvorttilt);
    zvorttiltVar.putVar(startp,countp,parcels->pclzvorttilt);
    xvortstretchVar.putVar(startp,countp,parcels->pclxvortstretch);
    yvortstretchVar.putVar(startp,countp,parcels->pclyvortstretch);
    zvortstretchVar.putVar(startp,countp,parcels->pclzvortstretch);
    cout << "*** SUCCESS writing file " << filename << "!" << endl;
    return;
}

#endif

