"""
Create an idealized domain, straight channel with a profile
and repeating tides.
"""

import stompy.model.delft.dflow_model as dfm
import stompy.model.suntans.sun_driver as sun
import os
import xarray as xr
import numpy as np
from stompy.grid import unstructured_grid
from stompy import utils
import matplotlib.pyplot as plt
import local_config
import six
six.moves.reload_module(local_config)
local_config.install()

##
# fabricate grid:
g=unstructured_grid.UnstructuredGrid(max_sides=4)
L=10000
W=75
g.add_rectilinear([0,0],[L,W],101,6)
g.add_node_field('z_bed',-(10-8*(2*g.nodes['x'][:,1]/W-1)**2))

#model=dfm.DFlowModel()
six.moves.reload_module(dfm)
six.moves.reload_module(sun)
local_config.install()
model=sun.SuntansModel()

model.set_grid(g)
model.run_start=np.datetime64("2010-01-01 00:00")
model.run_stop=np.datetime64("2010-01-05 00:00")

ds=xr.Dataset()
ds['time']=('time',),np.arange(model.run_start,model.run_stop,np.timedelta64(15,'m'))
t_sec=(ds.time-ds.time[0])/np.timedelta64(1,'s')
ds['eta']=('time',),np.cos( 2*np.pi * t_sec/(12*3600.))

model.add_bcs( dfm.StageBC(name='tide',z=ds.eta,geom=np.array( [[0,0],
                                                                [0,W]]) ) )

dt_secs=30
model.config['dt']=dt_secs
model.set_run_dir('channel')
model.sun_verbose_flag="-v"

# enable average output for ptm

model.config['ntout']=int(15*60/dt_secs) # 15 minutes
model.config['ntaverage']=int(15*60/dt_secs) # 15 minutes
model.config['calcaverage']=1
model.config['averageNetcdfFile']="average.nc"
# some kind of issue here where particles can't move between
# cells when it's 2D.  Run in 3D instead.
model.config['Nkmax']=10

model.write()
model.partition()
model.run_model()

##

# Convert to format for PTM
utils.path("/home/rusty/src")
import soda.dataio.ugrid.suntans2untrim
six.moves.reload_module(soda.dataio.ugrid.suntans2untrim)
from soda.dataio.ugrid.suntans2untrim import suntans2untrim

for avg_output in model.avg_outputs():
    ptm_output=os.path.join( os.path.dirname(avg_output),
                             "ptm-"+os.path.basename(avg_output) )
    suntans2untrim(avg_output,ptm_output,tstart=None,tend=None)


##
