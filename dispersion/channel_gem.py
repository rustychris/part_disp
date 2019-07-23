"""
Look into extracting dispersion rates from general exchange matrix
"""
import os
from stompy.model.fish_ptm import ptm_tools
from stompy.model.suntans import sun_driver
from stompy import utils
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import Bounds, minimize

##

hydro_dir="../hydro/channel"
ptm_dir="../ptm/uniform_channel"
ptm_bin=os.path.join(ptm_dir,"uniform_bin.out")

##

model=sun_driver.SuntansModel.load(hydro_dir)

pb=ptm_tools.PtmBin(ptm_bin)

##

ptm_bin_dt=pb.dt_seconds() # 900
hydro_period=12*3600 # 12 hour repeating tide

step0=0
stepN=step0+int(hydro_period/ptm_bin_dt)
d0,parts0=pb.read_timestep(0)
d,parts=pb.read_timestep(10)

# Get labels from an earlier timestep
labels=np.nan*np.zeros(parts0['id'].max()+1,np.float64)
labels[parts0['id']]=parts0['x'][:,0] # label by original x coordinate

x0=np.nan*np.zeros(parts0['id'].max()+1,np.float64)
x0[parts0['id']]=parts0['x'][:,0] 
y0=np.nan*np.zeros(parts0['id'].max()+1,np.float64)
y0[parts0['id']]=parts0['x'][:,1] 

## 
plt.figure(1).clf()
model.grid.plot_edges(color='k',lw=0.7)

plt.scatter(parts['x'][:,0],parts['x'][:,1],
            10,labels[parts['id']],
            cmap='jet')

##

fig=plt.figure(2)
fig.clf()
ax=fig.add_subplot(1,1,1)

ax.plot(labels[parts['id']], parts['x'][:,0], 'g.',ms=2)
ax.set_ylabel('x(t=T)')
ax.set_xlabel('x(t=0)')

##

df=pd.DataFrame()
df['x0']=x0[parts['id']]
df['y0']=y0[parts['id']]

df['xN']=parts['x'][:,0]
df['yN']=parts['x'][:,1]

##

if 1: # instead, build the aggregated grid directly, then assign cells
    g_agg=unstructured_grid.UnstructuredGrid(max_sides=10)
    bounds=model.grid.bounds()
    g_agg.add_rectilinear( [bounds[0],bounds[2]],
                           [bounds[1],bounds[3]],
                           nx=20,ny=2)
    Nc=g_agg.Ncells()
    e2c=g_agg.edge_to_cells()
    comp_edges=np.nonzero( e2c.min(axis=1)>=0 )[0]
    Nj=len(comp_edges)

def xy_to_bin(xy):
    if 0: 
        bins=np.linspace(0,10000,20) # boundaries between bins
        return np.searchsorted(bins,xy[:,0])-1
    if 1:
        shape=xy.shape[:-1]
        assert xy.shape[-1]==2
        rav=xy.reshape([-1,2])
    
        idxs=np.array( [ g_agg.select_cells_nearest(pnt,inside=True)
                         for pnt in rav] )
        return idxs.reshape(shape)
    
# Transition matrix
    
df['b0']=xy_to_bin( df.loc[:,['x0','y0']].values )
df['bN']=xy_to_bin( df.loc[:,['xN','yN']].values )

## 
M=np.zeros( (Nbins-1,Nbins-1), np.float64)

for idx,row in df.iterrows():
    M[int(row['b0']),
      int(row['bN'])] += 1

# normalize those by initial masses
bins0=np.searchsorted(bins,parts0['x'][:,0]) - 1
# initial mass in each bin
mass0=np.array( [ len(idxs) for _,idxs in utils.enumerate_groups(bins0)] )
# first column normalized by first mass, second by second etc.
M[:,:] *= (1./mass0)[None,:]

## 

fig=plt.figure(3)
fig.clf()
fig,ax=plt.subplots(1,1,num=3)
img=ax.imshow(M)
plt.colorbar(img)

##

# that looks right.
# from here,
# A. what matrix gives the instantaneous rates, such that
#   Q^(12 h) = M

# for that matter, how does fractional matrix exponentiation work (i.e. a matrix root?)
# almost certainly requires that the matrix is invertible.
# also what about having multiple roots?  like 1 has two square roots, +-1.
# this gets crazy many for matrices, though we can probably reign that in with
# the form of advection and diffusion operators.

# B. what set of per-edge advection and dispersion coefficients
#    approximate M when integrated over 24 hours?

##

# Extract the aggregated grid
if 0: # Old code glued original cells together to get aggregated cells.
    cell_bins=np.searchsorted(bins,model.grid.cells_center()[:,0]) - 1
    g_agg=unstructured_grid.UnstructuredGrid(max_sides=50)

    from shapely import ops
    for key,idxs in utils.enumerate_groups(cell_bins):
        polys=[model.grid.cell_polygon(c) for c in idxs]
        poly=ops.cascaded_union(polys).simplify(0.0)
        pnts=np.array(poly.exterior)[:-1]

        # also check for ordering - force CCW.
        if utils.signed_area(pnts)<0:
            pnts=pnts[::-1]

        nodes=[g_agg.add_or_find_node(x=x)
               for x in pnts]
        c=g_agg.add_cell_and_edges(nodes=nodes)
        assert c==key
        
## 

plt.figure(1).clf()
model.grid.plot_edges(lw=0.5,color='k',alpha=0.1)
g_agg.plot_edges(lw=0.5,color='k',alpha=0.7)
plt.axis('equal')
model.grid.plot_cells(values=cell_bins,cmap='jet')

##

e2c=g_agg.edge_to_cells()

##

# M_agg=np.zeros( (g_agg.Ncells(),g_agg.Ncells()), np.float64)


def K_to_D(K):
    D=np.zeros((Nc,Nc),np.float64)

    for j,k in enumerate(K):
        nc1=j   # true in toy example
        nc2=j+1
        assert k>=0
        D[nc1,nc1]-=k
        D[nc1,nc2]+=k
        D[nc2,nc1]+=k
        D[nc2,nc2]-=k
    return D
    
def Q_to_A(Q):
    A=np.zeros((Nc,Nc),np.float64)

    for j,q in enumerate(Q):
        nc1=j   # true in toy example
        nc2=j+1
        if q>=0:
            # flow is from nc1 to nc2, carrying nc1 concentration
            A[nc1,nc1]-=q
            A[nc2,nc1]+=q
        else:
            A[nc1,nc2]-=q
            A[nc2,nc2]+=q
    return A



def vec_to_M(vec):
    K=vec[:Nj]
    Q=vec[Nj:]
    return params_to_M(K=K,Q=Q)
def params_to_M(K,Q):
    return np.eye(Nc) + K_to_D(K) + Q_to_A(Q)


# Q[i] is the signed flow from Q[i] to Q[i+1]
Q=np.zeros(Nj,np.float64)

# K[i] is the dispersion volume between Q[i] and Q[i+1]
K=np.zeros(Nj,np.float64)
M_agg=params_to_M(K,Q)

##   

# start manual
vec=np.array( [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
               0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
               0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4])
## 

vec0=np.array( [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
               0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,
               0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4])

# 1 => 1.27
# 2 => 0.66
# 3 => 0.49
# 4 => 0.47
# 5 => 0.47
npow=3
def vec_to_op(vec):
    M_op=M_vec=vec_to_M(vec)
    for _ in range(npow-1):
        M_op=np.dot(M_op,M_vec)
    return M_op

def cost(vec):
    M_op=vec_to_op(vec)
    return np.linalg.norm(M-M_op)
from scipy.optimize import fmin

bounds= Bounds( [0]*Nj + [-1]*Nj,
                [1]*Nj + [1]*Nj )

res = minimize(cost, vec0, method='trust-constr', 
               options={'verbose': 1}, bounds=bounds)
vec=res['x']

# 
M_op=vec_to_op(vec)
plt.figure(4).clf()
fig,axs=plt.subplots(1,3,sharex=True,sharey=True,num=4)
axs[0].imshow(M,vmin=0,vmax=1.0)
axs[1].imshow(M_op,vmin=0,vmax=1.0)
axs[2].imshow(M_op-M,vmin=-1,vmax=1.0,cmap='seismic')

print( np.linalg.norm(M-M_op))

##

# display those advective and dispersive values on the grid
# edges

# What advection and dispersion coefficients come out of that?
K=vec[:Nj]
Q=vec[Nj:]


plt.figure(5).clf()
g_agg.plot_edges(lw=0.5,color='k',alpha=0.7)
g_agg.plot_edges(lw=1,mask=comp_edges,labeler='id')
#g_agg.plot_cells(lw=1.0,ec='r',alpha=0.5,labeler='id')
#g_agg.plot_nodes(labeler='id')


plt.axis('equal')



