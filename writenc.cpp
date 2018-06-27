#ifndef WRITENC_CPP
#define WRITENC_CPP

#include <iostream>
#include <string>
#include <netcdf>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

 
void initnc(string filename, int nParcels, int nTimes) {
    // Create the file.
    NcFile output(filename, NcFile::replace);

    NcDim pclDim = output.addDim("nParcels", nParcels);
    NcDim timeDim = output.addDim("nTimes", nTimes);
    //NcDim lvlDim = test.addDim(LVL_NAME, NLVL);
    //NcDim latDim = test.addDim(LAT_NAME, NLAT);
    //NcDim lonDim = test.addDim(LON_NAME, NLON);
    //NcDim recDim = test.addDim(REC_NAME);

    // define the coordinate variables
    //std::vector<NcDim> gridDimVector;
    //gridDimVector.push_back(latDim);
    //gridDimVector.push_back(lonDim);
    //NcVar latVar = test.addVar(LAT_NAME, ncFloat, gridDimVector);
    //NcVar lonVar = test.addVar(LON_NAME, ncFloat, gridDimVector);

    // Define the units attributes for coordinate vars. This
    // attatches a test attribute to each of the coordinate 
    // cariables containing the units
    //latVar.putAtt(UNITS, DEGREES_NORTH);
    //lonVar.putAtt(UNITS, DEGREES_EAST);

    // Define the netCDF variables for the fields
    // we want to store
    vector<NcDim> dimVector;
    //dimVector.push_back(lvlDim);
    //dimVector.push_back(latDim);
    //dimVector.push_back(lonDim);
    //NcVar sfccape_var = test.addVar(SFCCAPE_NAME, ncFloat, dimVector);
    //NcVar sfccinh_var = test.addVar(SFCCINH_NAME, ncFloat, dimVector);
    //NcVar sfclcl_var = test.addVar(SFCLCL_NAME, ncFloat, dimVector);
    //NcVar sfclfc_var = test.addVar(SFCLFC_NAME, ncFloat, dimVector);


    // Write the coordinate variable data to the file
    // latVar.putVar(lats);
    // lonVar.putVar(lons);
    cout << "*** SUCCESS writing file " << filename << "!" << endl;
    return;
}

void write_data(string filename, string varname, float *var_data) {

    NcFile dataFile(filename, NcFile::write);
    NcVar dataVar; 
    dataVar = dataFile.getVar(varname);
    dataVar.putVar(var_data);

    return;

}
#endif

