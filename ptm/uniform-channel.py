import os
import glob
import six
from stompy.model.fish_ptm import ptm_tools, ptm_config
from stompy.model.suntans import sun_driver
import numpy as np

##

hydro_dir="../hydro/channel"
model=sun_driver.SuntansModel.load(hydro_dir)
model.load_bc_ds()

##
six.moves.reload_module(ptm_config)
six.moves.reload_module(ptm_tools)

class Config(ptm_config.PtmConfig):
    def add_behaviors(self):
        self.lines+=["""\
BEHAVIOR INFORMATION
   NBEHAVIOR_PROFILES = 0
   NBEHAVIORS = 0
"""]
    def add_release_timing(self):
        self.lines+=["""\
RELEASE TIMING INFORMATION
   NRELEASE_TIMING_SETS = 1
   -- release timing set 1 ---        
     RELEASE_TIMING_SET = 'once'
     INITIAL_RELEASE_TIME = '{rel_time_str}'
     RELEASE_TIMING = 'single'
     INACTIVATION_TIME = 'none'""".format(rel_time_str=self.rel_time_str)
          ]
        
cfg=Config()

cfg.rel_time=model.run_start+24*np.timedelta64(1,'h')
cfg.end_time=model.run_stop - np.timedelta64(3600,'s')
cfg.run_dir="uniform_channel"


poly=model.grid.boundary_polygon().simplify(1.0)

cfg.regions.append(["""    REGION ='grid'
    REGION_POLYGON_FILE = 'grid.pol'
"""])
ptm_tools.geom2pol(poly,os.path.join(cfg.run_dir,'grid.pol'))
    

release=["""\
   RELEASE_DISTRIBUTION_SET = 'uniform' 
   MIN_BED_ELEVATION_METERS = -99.
   MAX_BED_ELEVATION_METERS =  99. 
   HORIZONTAL_DISTRIBUTION = 'region'
   DISTRIBUTION_IN_REGION = 'cell' 
   CELL_RELEASE_TIMING = 'independent'
   PARTICLE_NUMBER_CALCULATION_BASIS = 'volume'
   VOLUME_PER_PARTICLE_CUBIC_METERS = 1000
   ZMIN_NON_DIM = 0.0
   ZMAX_NON_DIM = 1.0
   VERT_SPACING = 'uniform'"""]
    
cfg.releases.append(release)

# For each of the flow inputs, add up, down, neutral
group=["""\
     GROUP = 'uniform'
     RELEASE_DISTRIBUTION_SET = 'uniform'
     REGION = 'grid'
     RELEASE_TIMING_SET = 'once'
     PARTICLE_TYPE = 'none'
     BEHAVIOR_SET = 'none'
     OUTPUT_SET = '15min_output'
     OUTPUT_FILE_BASE = 'uniform'
        """]
cfg.groups.append(group)
        
cfg.clean()
cfg.write()


##       
if 0:            
    print("Running PTM")
    if 'LD_LIBRARY_PATH' in os.environ:
        del os.environ['LD_LIBRARY_PATH']
    pwd=os.getcwd()
    try:
        os.chdir(cfg.run_dir)
        subprocess.run(["/home/rusty/src/fish_ptm/PTM/FISH_PTM.exe"])
    finally:
        os.chdir(pwd)

