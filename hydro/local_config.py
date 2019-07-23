# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:17:17 2018

@author: rustyh
"""

import sys, os

if sys.platform=='win32':
    dimr="C:/Program Files (x86)/Deltares/Delft3D FM Suite 2018.01 HMWQ (1.4.4.39490)/plugins/DeltaShell.Dimr/kernels/x64"
    dfm_bin_dir=os.path.join(dimr,'dflowfm','bin')
    share_bin_dir=os.path.join(dimr,'share','bin')
    dflowfm=os.path.join(dfm_bin_dir,'dflowfm-cli.exe')
    mpiexec=os.path.join(share_bin_dir,'mpiexec.exe')
    mapmerge=os.path.join(dfm_bin_dir,'dfmoutput.exe')
    delwaq1=os.path.join(dimr,'dwaq','bin','delwaq1.exe')
    delwaq2=os.path.join(dimr,'dwaq','bin','delwaq2.exe')
    dwaq_proc=os.path.join(dimr,'dwaq','default','proc_def')

    dimr_paths=";".join([dfm_bin_dir, share_bin_dir])
    run_dir_root="E:/proj/CacheSlough/Modeling/DFM/runs"
    # this is important to avoid a password dialog where mpiexec
    # tries to get network access.
    mpi_args=["-localonly"]
else:
    dfm_bin_dir="/home/rusty/src/dfm/1.5.2/lnx64/bin/"
    os.environ['LD_LIBRARY_PATH']="/home/rusty/src/dfm/1.5.2/lnx64/lib/"
    share_bin_dir=dfm_bin_dir
    run_dir_root="runs"

    sun_bin_dir="/home/rusty/src/suntans/main"

def install():
    from stompy.model.delft import dflow_model
    from stompy.model.suntans import sun_driver
    
    dflow_model.DFlowModel.dfm_bin_dir=dfm_bin_dir
    dflow_model.DFlowModel.mpi_bin_dir=share_bin_dir
    if sys.platform=='win32':
        dflow_model.DFlowModel.dfm_bin_exe="dflowfm-cli.exe"
        dflow_model.DFlowModel.mpi_args=('-localonly',)
        
    sun_driver.SuntansModel.sun_bin_dir=sun_bin_dir
    
