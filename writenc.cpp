#ifndef WRITENC_CPP
#define WRITENC_CPP

#include "datastructs.h"
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
    NcVar uturbVar = output.addVar("uturb", ncFloat, gridDimVector);
    NcVar vturbVar = output.addVar("vturb", ncFloat, gridDimVector);
    NcVar wturbVar = output.addVar("wturb", ncFloat, gridDimVector);
    NcVar udiffVar = output.addVar("udiff", ncFloat, gridDimVector);
    NcVar vdiffVar = output.addVar("vdiff", ncFloat, gridDimVector);
    NcVar wdiffVar = output.addVar("wdiff", ncFloat, gridDimVector);
    NcVar kmhVar = output.addVar("kmh", ncFloat, gridDimVector);

    NcVar xvortVar = output.addVar("xvort", ncFloat, gridDimVector);
    NcVar yvortVar = output.addVar("yvort", ncFloat, gridDimVector);
    NcVar zvortVar = output.addVar("zvort", ncFloat, gridDimVector);
    NcVar xvorttiltVar = output.addVar("xvorttilt", ncFloat, gridDimVector);
    NcVar yvorttiltVar = output.addVar("yvorttilt", ncFloat, gridDimVector);
    NcVar zvorttiltVar = output.addVar("zvorttilt", ncFloat, gridDimVector);
    NcVar xvortstretchVar = output.addVar("xvortstretch", ncFloat, gridDimVector);
    NcVar yvortstretchVar = output.addVar("yvortstretch", ncFloat, gridDimVector);
    NcVar zvortstretchVar = output.addVar("zvortstretch", ncFloat, gridDimVector);
    NcVar xvortsolenoidVar = output.addVar("xvortsolenoid", ncFloat, gridDimVector);
    NcVar yvortsolenoidVar = output.addVar("yvortsolenoid", ncFloat, gridDimVector);
    NcVar zvortsolenoidVar = output.addVar("zvortsolenoid", ncFloat, gridDimVector);
    NcVar xvortturbVar = output.addVar("xvortturb", ncFloat, gridDimVector);
    NcVar yvortturbVar = output.addVar("yvortturb", ncFloat, gridDimVector);
    NcVar zvortturbVar = output.addVar("zvortturb", ncFloat, gridDimVector);
    NcVar xvortdiffVar = output.addVar("xvortdiff", ncFloat, gridDimVector);
    NcVar yvortdiffVar = output.addVar("yvortdiff", ncFloat, gridDimVector);
    NcVar zvortdiffVar = output.addVar("zvortdiff", ncFloat, gridDimVector);

    NcVar ppertVar = output.addVar("prespert", ncFloat, gridDimVector);
    NcVar qvpertVar = output.addVar("qvpert", ncFloat, gridDimVector);
    NcVar rhopertVar = output.addVar("rhopert", ncFloat, gridDimVector);
    NcVar thetapertVar = output.addVar("thetapert", ncFloat, gridDimVector);
    NcVar thrhopertVar = output.addVar("thrhopert", ncFloat, gridDimVector);

    NcVar pbarVar = output.addVar("presbar", ncFloat, gridDimVector);
    NcVar qvbarVar = output.addVar("qvbar", ncFloat, gridDimVector);
    NcVar rhobarVar = output.addVar("rhobar", ncFloat, gridDimVector);
    NcVar thetabarVar = output.addVar("thetabar", ncFloat, gridDimVector);
    NcVar thrhobarVar = output.addVar("thrhobar", ncFloat, gridDimVector);

    NcVar qcVar = output.addVar("qc", ncFloat, gridDimVector);
    NcVar qiVar = output.addVar("qi", ncFloat, gridDimVector);
    NcVar qsVar = output.addVar("qs", ncFloat, gridDimVector);
    NcVar qgVar = output.addVar("qg", ncFloat, gridDimVector);

    // Define the units attributes for coordinate vars. This
    // attatches a test attribute to each of the coordinate 
    // cariables containing the units
    xVar.putAtt("units", "meters");
    yVar.putAtt("units", "meters");
    zVar.putAtt("units", "meters");

    uVar.putAtt("units", "meters / second");
    vVar.putAtt("units", "meters / second");
    wVar.putAtt("units", "meters / second");
    uturbVar.putAtt("units", "meters / second^2");
    vturbVar.putAtt("units", "meters / second^2");
    wturbVar.putAtt("units", "meters / second^2");
    udiffVar.putAtt("units", "meters / second^2");
    vdiffVar.putAtt("units", "meters / second^2");
    wdiffVar.putAtt("units", "meters / second^2");
    kmhVar.putAtt("units", "Unknown");

    xvortVar.putAtt("units", "s^-1");
    yvortVar.putAtt("units", "s^-1");
    zvortVar.putAtt("units", "s^-1");
    xvorttiltVar.putAtt("units", "s^-2");
    yvorttiltVar.putAtt("units", "s^-2");
    zvorttiltVar.putAtt("units", "s^-2");
    xvortstretchVar.putAtt("units", "s^-2");
    yvortstretchVar.putAtt("units", "s^-2");
    zvortstretchVar.putAtt("units", "s^-2");
    xvortsolenoidVar.putAtt("units", "s^-2");
    yvortsolenoidVar.putAtt("units", "s^-2");
    zvortsolenoidVar.putAtt("units", "s^-2");
    xvortturbVar.putAtt("units", "s^-2");
    yvortturbVar.putAtt("units", "s^-2");
    zvortturbVar.putAtt("units", "s^-2");
    xvortdiffVar.putAtt("units", "s^-2");
    yvortdiffVar.putAtt("units", "s^-2");
    zvortdiffVar.putAtt("units", "s^-2");

    ppertVar.putAtt("units", "Pa");
    qvpertVar.putAtt("units", "g kg^-1");
    rhopertVar.putAtt("units", "kg m^-3");
    thetapertVar.putAtt("units", "K");
    thrhopertVar.putAtt("units", "K");

    pbarVar.putAtt("units", "Pa");
    qvbarVar.putAtt("units", "g kg^-1");
    rhobarVar.putAtt("units", "kg m^-3");
    thetabarVar.putAtt("units", "K");
    thrhobarVar.putAtt("units", "K");

    qcVar.putAtt("units", "g kg^-1");
    qiVar.putAtt("units", "g kg^-1");
    qsVar.putAtt("units", "g kg^-1");
    qgVar.putAtt("units", "g kg^-1");
}
 
void write_parcels(string filename, parcel_pos *parcels, int writeIters ) { 
    NcFile output(filename, NcFile::write);

    NcVar xVar = output.getVar("xpos");
    NcVar yVar = output.getVar("ypos");
    NcVar zVar = output.getVar("zpos");

    NcVar uVar = output.getVar("u");
    NcVar vVar = output.getVar("v");
    NcVar wVar = output.getVar("w");
    NcVar uturbVar = output.getVar("uturb");
    NcVar vturbVar = output.getVar("vturb");
    NcVar wturbVar = output.getVar("wturb");
    NcVar udiffVar = output.getVar("udiff");
    NcVar vdiffVar = output.getVar("vdiff");
    NcVar wdiffVar = output.getVar("wdiff");
    NcVar kmhVar = output.getVar("khh");


    NcVar xvortVar = output.getVar("xvort");
    NcVar yvortVar = output.getVar("yvort");
    NcVar zvortVar = output.getVar("zvort");
    NcVar xvorttiltVar = output.getVar("xvorttilt");
    NcVar yvorttiltVar = output.getVar("yvorttilt");
    NcVar zvorttiltVar = output.getVar("zvorttilt");
    NcVar xvortstretchVar = output.getVar("xvortstretch");
    NcVar yvortstretchVar = output.getVar("yvortstretch");
    NcVar zvortstretchVar = output.getVar("zvortstretch");
    NcVar xvortsolenoidVar = output.getVar("xvortsolenoid");
    NcVar yvortsolenoidVar = output.getVar("yvortsolenoid");
    NcVar zvortsolenoidVar = output.getVar("zvortsolenoid");
    NcVar xvortturbVar = output.getVar("xvortturb");
    NcVar yvortturbVar = output.getVar("yvortturb");
    NcVar zvortturbVar = output.getVar("zvortturb");
    NcVar xvortdiffVar = output.getVar("xvortdiff");
    NcVar yvortdiffVar = output.getVar("yvortdiff");
    NcVar zvortdiffVar = output.getVar("zvortdiff");

    NcVar ppertVar = output.getVar("prespert");
    NcVar qvpertVar = output.getVar("qvpert");
    NcVar rhopertVar = output.getVar("rhopert");
    NcVar thetapertVar = output.getVar("thetapert");
    NcVar thrhopertVar = output.getVar("thrhopert");

    NcVar pbarVar = output.getVar("presbar");
    NcVar qvbarVar = output.getVar("qvbar");
    NcVar rhobarVar = output.getVar("rhobar");
    NcVar thetabarVar = output.getVar("thetabar");
    NcVar thrhobarVar = output.getVar("thrhobar");

    NcVar qcVar = output.getVar("qc");
    NcVar qiVar = output.getVar("qi");
    NcVar qsVar = output.getVar("qs");
    NcVar qgVar = output.getVar("qg");

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
    uturbVar.putVar(startp,countp,parcels->pcluturb);
    vturbVar.putVar(startp,countp,parcels->pclvturb);
    wturbVar.putVar(startp,countp,parcels->pclwturb);
    udiffVar.putVar(startp,countp,parcels->pcludiff);
    vdiffVar.putVar(startp,countp,parcels->pclvdiff);
    wdiffVar.putVar(startp,countp,parcels->pclwdiff);

    ppertVar.putVar(startp,countp,parcels->pclppert);
    qvpertVar.putVar(startp,countp,parcels->pclqvpert);
    rhopertVar.putVar(startp,countp,parcels->pclrhopert);
    thetapertVar.putVar(startp,countp,parcels->pclthetapert);
    thrhopertVar.putVar(startp,countp,parcels->pclthrhopert);

    pbarVar.putVar(startp,countp,parcels->pclpbar);
    qvbarVar.putVar(startp,countp,parcels->pclqvbar);
    rhobarVar.putVar(startp,countp,parcels->pclrhobar);
    thetabarVar.putVar(startp,countp,parcels->pclthetabar);
    thrhobarVar.putVar(startp,countp,parcels->pclthrhobar);

    qcVar.putVar(startp,countp,parcels->pclqc);
    qiVar.putVar(startp,countp,parcels->pclqi);
    qsVar.putVar(startp,countp,parcels->pclqs);
    qgVar.putVar(startp,countp,parcels->pclqg);

    xvortVar.putVar(startp,countp,parcels->pclxvort);
    yvortVar.putVar(startp,countp,parcels->pclyvort);
    zvortVar.putVar(startp,countp,parcels->pclzvort);
    xvorttiltVar.putVar(startp,countp,parcels->pclxvorttilt);
    yvorttiltVar.putVar(startp,countp,parcels->pclyvorttilt);
    zvorttiltVar.putVar(startp,countp,parcels->pclzvorttilt);
    xvortstretchVar.putVar(startp,countp,parcels->pclxvortstretch);
    yvortstretchVar.putVar(startp,countp,parcels->pclyvortstretch);
    zvortstretchVar.putVar(startp,countp,parcels->pclzvortstretch);
    xvortsolenoidVar.putVar(startp,countp,parcels->pclxvortsolenoid);
    yvortsolenoidVar.putVar(startp,countp,parcels->pclyvortsolenoid);
    zvortsolenoidVar.putVar(startp,countp,parcels->pclzvortsolenoid);
    xvortturbVar.putVar(startp,countp,parcels->pclxvortturb);
    yvortturbVar.putVar(startp,countp,parcels->pclyvortturb);
    zvortturbVar.putVar(startp,countp,parcels->pclzvortturb);
    xvortdiffVar.putVar(startp,countp,parcels->pclxvortdiff);
    yvortdiffVar.putVar(startp,countp,parcels->pclyvortdiff);
    zvortdiffVar.putVar(startp,countp,parcels->pclzvortdiff);
    cout << "*** SUCCESS writing file " << filename << "!" << endl;
    return;
}

#endif

