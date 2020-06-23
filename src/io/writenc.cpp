#ifndef WRITENC_CPP
#define WRITENC_CPP
/*
 * Copyright (C) 2017-2020 Kelton Halbert, Space Science and Engineering Center (SSEC), University of Wisconsin - Madison
 * Written by Kelton Halbert at the University of Wisconsin - Madison,
 * Cooperative Institute for Meteorological Satellite Studies (CIMSS),
 * Space Science and Engineering Center (SSEC). Provided under the Apache 2.0 License.
 * Email: kthalbert@wisc.edu
*/

extern "C" {
#include <lofs-read.h>
#include <lofs-dirstruct.h>
#include <lofs-hdf2nc.h>
#include <lofs-limits.h>
#include <lofs-macros.h>
}
#include "../include/datastructs.h"
#include <iostream>
#include <string>
#include <netcdf>

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

void init_nc(string filename, parcel_pos *parcels) {
    // get the io configurations from the user
    iocfg *io = parcels->io;
    // Create the file.
    NcFile output(filename, NcFile::replace);

    NcDim pclDim = output.addDim("nParcels", parcels->nParcels);
    NcDim timeDim = output.addDim("nTimes");

	vector<NcDim> timeDimVector;
	timeDimVector.push_back(timeDim);
	NcVar timeVar = output.addVar("time", ncFloat, timeDimVector);
	timeVar.putAtt("units", "seconds");
	timeVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);

    // define the coordinate variables
    vector<NcDim> gridDimVector;
    gridDimVector.push_back(pclDim);
    gridDimVector.push_back(timeDim);
    NcVar xVar = output.addVar("xpos", ncFloat, gridDimVector);
    NcVar yVar = output.addVar("ypos", ncFloat, gridDimVector);
    NcVar zVar = output.addVar("zpos", ncFloat, gridDimVector);
    xVar.putAtt("units", "meters");
    yVar.putAtt("units", "meters");
    zVar.putAtt("units", "meters");
	xVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
	yVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
	zVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);


    NcVar uVar = output.addVar("u", ncFloat, gridDimVector);
    NcVar vVar = output.addVar("v", ncFloat, gridDimVector);
    NcVar wVar = output.addVar("w", ncFloat, gridDimVector);
    uVar.putAtt("units", "meters / second");
    vVar.putAtt("units", "meters / second");
    wVar.putAtt("units", "meters / second");
	uVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
	vVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
	wVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);

    if (io->output_momentum_budget) {
        NcVar wbuoyVar = output.addVar("wbuoy", ncFloat, gridDimVector);
        NcVar upgradVar = output.addVar("upgrad", ncFloat, gridDimVector);
        NcVar vpgradVar = output.addVar("vpgrad", ncFloat, gridDimVector);
        NcVar wpgradVar = output.addVar("wpgrad", ncFloat, gridDimVector);
        NcVar uturbVar = output.addVar("uturb", ncFloat, gridDimVector);
        NcVar vturbVar = output.addVar("vturb", ncFloat, gridDimVector);
        NcVar wturbVar = output.addVar("wturb", ncFloat, gridDimVector);
        NcVar udiffVar = output.addVar("udiff", ncFloat, gridDimVector);
        NcVar vdiffVar = output.addVar("vdiff", ncFloat, gridDimVector);
        NcVar wdiffVar = output.addVar("wdiff", ncFloat, gridDimVector);

        wbuoyVar.putAtt("units", "meters / second^2");
		wbuoyVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        upgradVar.putAtt("units", "meters / second^2");
		upgradVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        vpgradVar.putAtt("units", "meters / second^2");
		vpgradVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        wpgradVar.putAtt("units", "meters / second^2");
		wpgradVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        uturbVar.putAtt("units", "meters / second^2");
		uturbVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        vturbVar.putAtt("units", "meters / second^2");
		vturbVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        wturbVar.putAtt("units", "meters / second^2");
		wturbVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        udiffVar.putAtt("units", "meters / second^2");
		udiffVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        vdiffVar.putAtt("units", "meters / second^2");
		vdiffVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        wdiffVar.putAtt("units", "meters / second^2");
		wdiffVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }

    if (io->output_kmh) {
        NcVar kmhVar = output.addVar("kmh", ncFloat, gridDimVector);
        kmhVar.putAtt("units", "Unknown");
		kmhVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }

    if (io->output_vorticity_budget || io->output_xvort) {
        NcVar xvortVar = output.addVar("xvort", ncFloat, gridDimVector);
        xvortVar.putAtt("units", "s^-1");
		xvortVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_vorticity_budget || io->output_yvort) {
        NcVar yvortVar = output.addVar("yvort", ncFloat, gridDimVector);
        yvortVar.putAtt("units", "s^-1");
		yvortVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_vorticity_budget || io->output_zvort) {
        NcVar zvortVar = output.addVar("zvort", ncFloat, gridDimVector);
        zvortVar.putAtt("units", "s^-1");
		zvortVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_vorticity_budget) {
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
        NcVar xvortbaroVar = output.addVar("xvortbaro", ncFloat, gridDimVector);
        NcVar yvortbaroVar = output.addVar("yvortbaro", ncFloat, gridDimVector);

        xvorttiltVar.putAtt("units", "s^-2");
		xvorttiltVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        yvorttiltVar.putAtt("units", "s^-2");
		yvorttiltVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        zvorttiltVar.putAtt("units", "s^-2");
		zvorttiltVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        xvortstretchVar.putAtt("units", "s^-2");
		xvortstretchVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        yvortstretchVar.putAtt("units", "s^-2");
		yvortstretchVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        zvortstretchVar.putAtt("units", "s^-2");
		zvortstretchVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        xvortsolenoidVar.putAtt("units", "s^-2");
		xvortsolenoidVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        yvortsolenoidVar.putAtt("units", "s^-2");
		yvortsolenoidVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        zvortsolenoidVar.putAtt("units", "s^-2");
		zvortsolenoidVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        xvortturbVar.putAtt("units", "s^-2");
		xvortturbVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        yvortturbVar.putAtt("units", "s^-2");
		yvortturbVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        zvortturbVar.putAtt("units", "s^-2");
		zvortturbVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        xvortdiffVar.putAtt("units", "s^-2");
		xvortdiffVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        yvortdiffVar.putAtt("units", "s^-2");
		yvortdiffVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        zvortdiffVar.putAtt("units", "s^-2");
		zvortdiffVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        xvortbaroVar.putAtt("units", "s^-2");
		xvortbaroVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
        yvortbaroVar.putAtt("units", "s^-2");
		yvortbaroVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }

    if (io->output_ppert) {
        NcVar ppertVar = output.addVar("prespert", ncFloat, gridDimVector);
        ppertVar.putAtt("units", "Pa");
		ppertVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_qvpert) {
        NcVar qvpertVar = output.addVar("qvpert", ncFloat, gridDimVector);
        qvpertVar.putAtt("units", "g kg^-1");
		qvpertVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_rhopert) {
        NcVar rhopertVar = output.addVar("rhopert", ncFloat, gridDimVector);
        rhopertVar.putAtt("units", "kg m^-3");
		rhopertVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_thetapert) {
        NcVar thetapertVar = output.addVar("thetapert", ncFloat, gridDimVector);
        thetapertVar.putAtt("units", "K");
		thetapertVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_thrhopert) {
        NcVar thrhopertVar = output.addVar("thrhopert", ncFloat, gridDimVector);
        thrhopertVar.putAtt("units", "K");
		thrhopertVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }

    if (io->output_pbar) {
        NcVar pbarVar = output.addVar("presbar", ncFloat, gridDimVector);
        pbarVar.putAtt("units", "Pa");
		pbarVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_qvbar) {
        NcVar qvbarVar = output.addVar("qvbar", ncFloat, gridDimVector);
        qvbarVar.putAtt("units", "g kg^-1");
		qvbarVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_rhobar) {
        NcVar rhobarVar = output.addVar("rhobar", ncFloat, gridDimVector);
        rhobarVar.putAtt("units", "kg m^-3");
		rhobarVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_thetabar) {
        NcVar thetabarVar = output.addVar("thetabar", ncFloat, gridDimVector);
        thetabarVar.putAtt("units", "K");
		thetabarVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_thrhobar) {
        NcVar thrhobarVar = output.addVar("thrhobar", ncFloat, gridDimVector);
        thrhobarVar.putAtt("units", "K");
		thrhobarVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }

    if (io->output_qc) {
        NcVar qcVar = output.addVar("qc", ncFloat, gridDimVector);
        qcVar.putAtt("units", "g kg^-1");
		qcVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_qi) {
        NcVar qiVar = output.addVar("qi", ncFloat, gridDimVector);
        qiVar.putAtt("units", "g kg^-1");
		qiVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_qs) {
        NcVar qsVar = output.addVar("qs", ncFloat, gridDimVector);
        qsVar.putAtt("units", "g kg^-1");
		qsVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_qg) {
        NcVar qgVar = output.addVar("qg", ncFloat, gridDimVector);
        qgVar.putAtt("units", "g kg^-1");
		qgVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
    if (io->output_qr) {
        NcVar qrVar = output.addVar("qr", ncFloat, gridDimVector);
        qrVar.putAtt("units", "g kg^-1");
		qrVar.putAtt("_FillValue", ncFloat, NC_FILL_FLOAT);
    }
}
 
void write_parcels(string filename, parcel_pos *parcels, int writeIters ) { 
    // get the io configurations from the user
    iocfg *io = parcels->io;
    // These vectors define the starting write positions
    // and the number of bits to write
    vector<size_t> startp,countp;
	vector<size_t> startt,countt;

	startt.push_back(0);
	if (writeIters == 0) startt.push_back(0);
	else startt.push_back(parcels->nTimes * writeIters - writeIters);
	countt.push_back(parcels->nTimes);

    startp.push_back(0);
    if (writeIters == 0) startp.push_back(0);
    else startp.push_back(parcels->nTimes * writeIters - writeIters);
    countp.push_back(parcels->nParcels);
    countp.push_back(parcels->nTimes);

    // open the file for writing
    NcFile output(filename, NcFile::write);

    NcVar timeVar = output.getVar("time");
	timeVar.putVar(startt, countt, parcels->time);

    NcVar xVar = output.getVar("xpos");
    NcVar yVar = output.getVar("ypos");
    NcVar zVar = output.getVar("zpos");
    xVar.putVar(startp,countp,parcels->xpos);
    yVar.putVar(startp,countp,parcels->ypos);
    zVar.putVar(startp,countp,parcels->zpos);

    NcVar uVar = output.getVar("u");
    NcVar vVar = output.getVar("v");
    NcVar wVar = output.getVar("w");
    uVar.putVar(startp,countp,parcels->pclu);
    vVar.putVar(startp,countp,parcels->pclv);
    wVar.putVar(startp,countp,parcels->pclw);

    if (io->output_momentum_budget) {
        NcVar wbuoyVar = output.getVar("wbuoy");
        NcVar upgradVar = output.getVar("upgrad");
        NcVar vpgradVar = output.getVar("vpgrad");
        NcVar wpgradVar = output.getVar("wpgrad");
        NcVar uturbVar = output.getVar("uturb");
        NcVar vturbVar = output.getVar("vturb");
        NcVar wturbVar = output.getVar("wturb");
        NcVar udiffVar = output.getVar("udiff");
        NcVar vdiffVar = output.getVar("vdiff");
        NcVar wdiffVar = output.getVar("wdiff");

        wbuoyVar.putVar(startp,countp,parcels->pclbuoy);
        upgradVar.putVar(startp,countp,parcels->pclupgrad);
        vpgradVar.putVar(startp,countp,parcels->pclvpgrad);
        wpgradVar.putVar(startp,countp,parcels->pclwpgrad);
        uturbVar.putVar(startp,countp,parcels->pcluturb);
        vturbVar.putVar(startp,countp,parcels->pclvturb);
        wturbVar.putVar(startp,countp,parcels->pclwturb);
        udiffVar.putVar(startp,countp,parcels->pcludiff);
        vdiffVar.putVar(startp,countp,parcels->pclvdiff);
        wdiffVar.putVar(startp,countp,parcels->pclwdiff);
    }
    if (io->output_kmh) {
        NcVar kmhVar = output.getVar("kmh");
        kmhVar.putVar(startp,countp,parcels->pclkmh);
    }


    if (io->output_vorticity_budget || io->output_xvort) {
        NcVar xvortVar = output.getVar("xvort");
        xvortVar.putVar(startp,countp,parcels->pclxvort);
    }
    if (io->output_vorticity_budget || io->output_yvort) {
        NcVar yvortVar = output.getVar("yvort");
        yvortVar.putVar(startp,countp,parcels->pclyvort);
    }
    if (io->output_vorticity_budget || io->output_zvort) {
        NcVar zvortVar = output.getVar("zvort");
        zvortVar.putVar(startp,countp,parcels->pclzvort);
    }
    if (io->output_vorticity_budget) {
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
        NcVar xvortbaroVar = output.getVar("xvortbaro");
        NcVar yvortbaroVar = output.getVar("yvortbaro");

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
        xvortbaroVar.putVar(startp,countp,parcels->pclxvortbaro);
        yvortbaroVar.putVar(startp,countp,parcels->pclyvortbaro);
    }

    if (io->output_ppert) {
        NcVar ppertVar = output.getVar("prespert");
        ppertVar.putVar(startp,countp,parcels->pclppert);
    }
    if (io->output_qvpert) {
        NcVar qvpertVar = output.getVar("qvpert");
        qvpertVar.putVar(startp,countp,parcels->pclqvpert);
    }
    if (io->output_rhopert) {
        NcVar rhopertVar = output.getVar("rhopert");
        rhopertVar.putVar(startp,countp,parcels->pclrhopert);
    }
    if (io->output_thetapert) {
        NcVar thetapertVar = output.getVar("thetapert");
        thetapertVar.putVar(startp,countp,parcels->pclthetapert);
    }
    if (io->output_thrhopert) {
        NcVar thrhopertVar = output.getVar("thrhopert");
        thrhopertVar.putVar(startp,countp,parcels->pclthrhopert);
    }

    if (io->output_pbar) {
        NcVar pbarVar = output.getVar("presbar");
        pbarVar.putVar(startp,countp,parcels->pclpbar);
    }
    if (io->output_qvbar) {
        NcVar qvbarVar = output.getVar("qvbar");
        qvbarVar.putVar(startp,countp,parcels->pclqvbar);
    }
    if (io->output_rhobar) {
        NcVar rhobarVar = output.getVar("rhobar");
        rhobarVar.putVar(startp,countp,parcels->pclrhobar);
    }
    if (io->output_thetabar) {
        NcVar thetabarVar = output.getVar("thetabar");
        thetabarVar.putVar(startp,countp,parcels->pclthetabar);
    }
    if (io->output_thrhobar) {
        NcVar thrhobarVar = output.getVar("thrhobar");
        thrhobarVar.putVar(startp,countp,parcels->pclthrhobar);
    }

    if (io->output_qc) {
        NcVar qcVar = output.getVar("qc");
        qcVar.putVar(startp,countp,parcels->pclqc);
    }
    if (io->output_qi) {
        NcVar qiVar = output.getVar("qi");
        qiVar.putVar(startp,countp,parcels->pclqi);
    }
    if (io->output_qs) {
        NcVar qsVar = output.getVar("qs");
        qsVar.putVar(startp,countp,parcels->pclqs);
    }
    if (io->output_qg) {
        NcVar qgVar = output.getVar("qg");
        qgVar.putVar(startp,countp,parcels->pclqg);
    }
    if (io->output_qr) {
        NcVar qrVar = output.getVar("qr");
        qrVar.putVar(startp,countp,parcels->pclqr);
    }

    return;
}

#endif

