## This script generates a curves for fluid particle trajectories
## This is intended as the script of a 'Programmable Source'
## Author: Kelton Halbert
## Institution: University of Wisconsin - Madison 
## Department: Atmospheric and Oceanic Sciences
## Research Group: Cooperative Institute for Meteorological Satellite Studies
## Date: Oct 29, 2019
import numpy as np

pdi = self.GetInput()
pdo = self.GetPolyDataOutput()

## get nParcels and nTimes from
## the input dataset
bounds = pdi.GetBounds()
nParcels = int(bounds[3]+1)
nTimes = int(bounds[1]+1)

## The names of the arrays containing position
## information about the parcels
poskeys = ["xpos", "ypos", "zpos"]
vars = inputs[0].PointData.keys()

varcount = 0
for var in vars:
	if var not in poskeys:
		newArray = vtk.vtkFloatArray()
		newArray.SetName(var)
		newArray.SetNumberOfComponents(1)
		oldArray = inputs[0].PointData[var]
		array_out = newArray
		print(var, array_out)

		## copy the values into the output array
		for i in range(len(oldArray)):
			array_out.InsertNextValue(oldArray[i])
		pdo.GetPointData().AddArray(newArray)
		varcount += 1
## Allocate the number of 'cells' that will be added. We are just
## adding one vtkPolyLine 'cell' to the vtkPolyData object.
pdo.Allocate(nParcels, 1)

## We divide the position by 1000 to get into units of km since
## our 3D data is stored in units of km
xpos = inputs[0].PointData[poskeys[0]].reshape((nParcels, nTimes)) / 1000.
ypos = inputs[0].PointData[poskeys[1]].reshape((nParcels, nTimes)) / 1000.
zpos = inputs[0].PointData[poskeys[2]].reshape((nParcels, nTimes)) / 1000.
## This will store the points for the parcel trajectory
newPts = vtk.vtkPoints()


## Loop over our parcels
for pcl in range(0, nParcels):
	## Loop over each time step per parcel
    for i in range(0, nTimes):
       ## Generate the Points along the parcel curve
       x = xpos[pcl, i]
       y = ypos[pcl, i]
       z = zpos[pcl, i]

       ## Insert the Points into the vtkPoints object
       ## The first parameter indicates the reference.
       ## value for the point. The reference value is
       ## offset so that we keep each parcel
       ## seperate from each other. 
       newPts.InsertPoint(i+nTimes*pcl, x, y, z)

    ## Add the points to the vtkPolyData object
    ## Right now the points are not associated with a line - 
    ## it is just a set of unconnected points. We need to
    ## create a 'cell' object that ties points together
    ## to make a curve (in this case). This is done below.
    ## A 'cell' is just an object that tells how points are
    ## connected to make a 1D, 2D, or 3D object.
    pdo.SetPoints(newPts)

    ## Make a vtkPolyLine which holds the info necessary
    ## to create a curve composed of line segments. This
    ## really just hold constructor data that will be passed
    ## to vtkPolyData to add a new line.
    aPolyLine = vtk.vtkPolyLine()

    #Indicate the number of points along the line
    aPolyLine.GetPointIds().SetNumberOfIds(nTimes)

    ## Now we loop over our times again, this time
    ## Relating each parcel trace to a line rather
    ## than unrelated points. 
    for i in range(0,nTimes):
       ## Add the points to the line. The first value indicates
       ## the order of the point on the line. The second value
       ## is a reference to a point in a vtkPoints object. Depends
       ## on the order that Points were added to vtkPoints object.
       ## Note that this will not be associated with actual points
       ## until it is added to a vtkPolyData object which holds a
       ## vtkPoints object.
       aPolyLine.GetPointIds().SetId(i, i+nTimes*pcl)

    ## Add the poly line 'cell' to the vtkPolyData object.

    pdo.InsertNextCell(aPolyLine.GetCellType(), aPolyLine.GetPointIds())

    ## The trajectories are ready to plot! Click 'Apply'.