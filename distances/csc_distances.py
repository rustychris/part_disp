"""
Calculate several distance along axis metrics for the CSC 
grid and compare.
"""
import six
import numpy as np
import matplotlib.pyplot as plt

from stompy.grid import unstructured_grid
from stompy import utils
from sklearn.linear_model import LinearRegression
import stompy.plot.cmap as scmap
from stompy.plot import plot_utils
cmap=scmap.load_gradient('ncview_banded.cpt')

## 

grid_file="~/src/csc/dflowfm/CacheSloughComplex_v111-edit19fix-bathy.nc"
six.moves.reload_module(unstructured_grid)

g=unstructured_grid.UnstructuredGrid.read_ugrid(grid_file)

ds=g.write_to_xarray()

##

# Define a downstream d=0
x_down=np.array([610016, 4215971.])

##
from stompy.model import unstructured_diffuser
six.moves.reload_module(unstructured_diffuser)

##

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)
g.plot_edges(lw=0.5,color='k',ax=ax)
ncoll=g.plot_nodes(values=g.nodes['depth'],ax=ax,cmap='jet')
plt.colorbar(ncoll)

# node depths are positive up.

##
def fig_dist(C,num=2,log=False,title="",local_max=False, direction=False):
    fig=plt.figure(num)
    fig.clf()
    fig.set_size_inches([6,9],forward=True)
    ax=fig.add_subplot(1,1,1)
    cax=fig.add_axes([0.05,0.6,0.03,0.35])
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    if log:
        C=np.log10(C.clip(1e-10,np.inf))
        label='log$_{10}$'
    else:
        label='linear'
        
    ccoll=g.plot_cells(values=C,ax=ax,cmap=cmap)
    ccoll.set_lw(0.05)
    ccoll.set_edgecolor('face')
    
    # plt.colorbar(ccoll,cax=cax,label=label)
    plot_utils.cbar(ccoll,cax=cax,label=label)
    
    if local_max:
        is_local_max=np.ones(g.Ncells(),np.bool8)
        e2c=g.edge_to_cells()
        internal=e2c.min(axis=1)>=0
        c1=e2c[internal,0]
        c2=e2c[internal,1]
        c1_less=C[c1]<C[c2]
        is_local_max[ c1[c1_less] ]=False
        c2_less=C[c2]<C[c1]
        is_local_max[ c2[c2_less] ]=False
        cc=g.cells_center()
        ax.plot(cc[is_local_max,0],cc[is_local_max,1],'ko')

    if direction:
        nbrhood=4
        idxs=np.arange(g.Ncells())
        np.random.shuffle(idxs)
        samp_cells=idxs[:1000]
        cc=g.cells_center()
        XY=cc[samp_cells]
        UV=np.nan*XY
        for i,c in utils.progress(enumerate(samp_cells)):
            cells=[c]
            for _ in range(nbrhood): cells=np.unique(list(cells) + [c for c0 in cells for c in g.cell_to_cells(c0) if c>=0] )
            #y=C[cells]
            #X=np.c_[cc[cells]-cc[cells].mean(axis=0),
            #        np.ones(len(cells))] # location and bias
            #beta_hat=np.linalg.lstsq(X,y,rcond=None)[0]
            #UV[i,:]=beta_hat[:2] # [gradx,grady]
            low=cells[ np.argmin(C[cells]) ]
            high=cells[ np.argmax(C[cells]) ]
            UV[i,:]=(cc[high]-cc[low])/(C[high]-C[low])
        ax.quiver( XY[:,0], XY[:,1], UV[:,0], UV[:,1],pivot='tip',scale=60,width=0.005)
        
    ax.xaxis.set_visible(0)
    ax.yaxis.set_visible(0)
    ax.axis('equal')
    ax.text(0.5,0.98,title,transform=ax.transAxes,va='top',ha='center')
    return fig

fig=fig_dist(cell_costs,num=6,log=False,title='Dijkstra',local_max=False, direction=True)

##


def cost_to_dist(C,g=g,monotonic_search=True):
    """
    given some cost metric C on the cells of the grid,
    use the shortest path from C.min() to C.max() to translate
    cost C into geographic distance.

    monotonic_search: shortest path only traverses edges of increasing
    cost.  Fails if there are local maxima along all paths between
    global min/max.
    """
    c_min=np.argmin(C)
    c_max=np.argmax(C)
    e2c=g.edge_to_cells()
    
    edge_selector=lambda j,direc: (direc*C[e2c[j,0]]<direc*C[e2c[j,1]])

    path=g.shortest_path(c_min, c_max, traverse='cells',edge_selector=edge_selector)
    
    path_xy=g.cells_centroid()[path]
    path_dists=utils.dist_along(path_xy)
    path_C=C[path]

    path_C_orig=path_C.copy()
    path_C.sort() # minor fudging, but this needs to be monotonic
    non_monotonic=np.std(path_C_orig-path_C)/(C[c_max]-C[c_min])
    print("Shortest path deviated from monotonic cost by %e"%non_monotonic)

    dist=np.interp(C, path_C,path_dists)
    return dist

## 
# Diffusion distance from x_down, no bathymetry.
# unit flux into the origin, and spatially constant decay rate

gd=unstructured_diffuser.Diffuser(g)
gd.set_flux(xy=x_down, value=10.0)
gd.set_decay_rate(1e-8)
gd.construct_linear_system()
C=gd.compute()

D=cost_to_dist(-C)
fig=fig_dist(D,num=2,log=False,title="Diffusion and decay, no bathy",local_max=True)

ds['dist_diff']=('face',),D
ds['dist_diff'].attrs['desc']="K=100, decay=1e-8, Decker flux=10"


##

# finite flux enters at the downstream end
# would like the gradient to be the same for the same cross
# cross sectional area.  so for a single channel, if it 
# has a deep constriction or a broad sill, dC/dx shouldn't
# be different.

edge_depth= -g.nodes['depth'][g.edges['nodes']].mean(axis=1).clip(-np.inf,-0.1)
gd=unstructured_diffuser.Diffuser(g,edge_depth=edge_depth)
gd.set_flux(xy=x_down, value=10.0)
gd.set_decay_rate(1e-8)
gd.construct_linear_system()
C=gd.compute()
fig=fig_dist(C,num=3,log=False)
fig.suptitle("Diffusion and decay, edge bathy")

##

edge_depth= -g.nodes['depth'][g.edges['nodes']].mean(axis=1).clip(-np.inf,-0.1)
cell_depth= -g.interp_node_to_cell(g.nodes['depth']).clip(-np.inf,-0.1)

gd=unstructured_diffuser.Diffuser(g,edge_depth=edge_depth,cell_depth=cell_depth)
gd.set_flux(xy=x_down, value=10.0)
gd.set_decay_rate(1e-8)
gd.construct_linear_system()
C=gd.compute()
fig=fig_dist(C,num=4,log=True)
fig.suptitle("Diffusion and decay, edge+cell bathy")

## 

# Almost want to ignore edge lengths l_j, just use d_j.  
# l_j always appears with dzf

# offset effect of edge length, but this introduces a resolution dependence, such
# that the same channel but with double the resolution gets double the Ayz.
edge_depth= 1./g.edges_length()

gd=unstructured_diffuser.Diffuser(g,edge_depth=edge_depth)
gd.set_flux(xy=x_down, value=10.0)
gd.set_decay_rate(1e-8)
gd.construct_linear_system()
C=gd.compute()
fig=fig_dist(C,num=5,log=True)
fig.suptitle("Diffusion and decay, edge=1/l")

##

# generally it's a problem that a local gradient is determined by
# the decay load upstream.
# at the branch of Cache Slough and DWS, there is much more demand
# upstream on the Cache Slough side, so even though it funnels through

##

# old-fashioned Dijkstra
# As in the past, this can lead to local maxima, such as on the mainstem
# Sac where it is faster to go up Steamboat, and down the Sac, than to
# go up the Sac the whole way.
c1=g.select_cells_nearest(x_down)
costs=g.shortest_path(c1,np.arange(g.Ncells()),return_type='cost',traverse='cells')
cells=np.array([c[0] for c in costs])
orig_costs=np.array([c[1] for c in costs])
cell_costs=np.nan*np.ones(g.Ncells())
cell_costs[cells]=orig_costs

fig=fig_dist(cell_costs,num=6,log=False,title='Dijkstra',local_max=False, direction=True)


ds['dist_dijk']=('face',),cell_costs
ds['dist_dijk'].attrs['desc']="Dijstra from Decker"

##

# In the absence of cycles, Dijkstra is good.
# In the presence of cycles, Dijkstra creates local maxima of distance.
# Diffusion generally avoids local maxima
#   - the nonlocal, upstream decay drives the direction of local gradients,
#     such that redundant paths that are part of a cycle will have gradients
#     driven by where they fall relative to the global distribution of decay.

# If diffusion were used to come up with the direction, and then dijkstra used
# on the resulting directed graph, we'd end up with discontinuities.

# Also generate some along-axis distances to Sac, Cache, Lindsey

x_lindsey=np.array([604534., 4235344.])
x_barker=np.array([604725., 4237333.])
x_ulatis=np.array([603697., 4243011])
x_hass=np.array([608498., 4242599.])
x_dws=np.array([627233., 4269089.])
x_sac=np.array([619931., 4293699.])


for name,dest in [('lindsey',x_lindsey),
                  ('barker',x_barker),
                  ('ulatis',x_ulatis),
                  ('hass',x_hass),
                  ('dws',x_dws),
                  ('sac',x_sac)]:
    print(name)
    gd=unstructured_diffuser.Diffuser(g)
    gd.set_dirichlet(xy=x_down, value=1.0)
    gd.set_dirichlet(xy=dest,value=0.0)
    gd.set_decay_rate(0.0) # just to be sure
    C=gd.compute()
    # in the past, the remapping to distance was specific to the
    # destination point, but that will have the highest diffusion
    # cost, so cost_to_dist() will figure out the right destination
    # point without any further information.
    D=cost_to_dist(C,g=g)
    vname='dist_diff_%s'%name
    ds[vname]=('face',),D
    ds[vname].attrs['desc']="Diffusion/dirichlet distance from Decker to %s"%name
    
## 
ds.to_netcdf('csc_distances.nc')

## 
#fig.axes[0].plot(path_xy[:,0],path_xy[:,1],'k-',zorder=3)

#fig=fig_dist(dist,num=2,log=False,title="Diffusion/euclidean Decker->Lindsey")

##

# Plot each of the distance fields and save to figure.

for v in ds.data_vars:
    if not v.startswith('dist'): continue
    print(v)

    fig=fig_dist(ds[v].values,log=False,title=ds[v].attrs['desc'],num=10)
    fig.savefig('map_%s.png'%v,dpi=200)

##

# And what does magnitude of distance gradient look like?
from stompy.model.stream_tracer import U_perot

e2c=g.edge_to_cells().copy()
cc=g.cells_center()
ec=g.edges_center()

c1=e2c[:,0]
c2=e2c[:,1]
c1[c1<0]=c2[c1<0]
c2[c2<0]=c1[c2<0]
dg=np.where(c1==c2,
            1.0,  # cell differences will always be zero, so this doesn't matter.
            utils.dist(cc[c1] - cc[c2]))

def dist_ratio(D):
    # calculate per-edge gradients
    dDdn=(D[c2]-D[c1])/dg
    gradD=U_perot(g,g.edges_length()*dDdn,g.cells_area())
    gradmag=utils.mag(gradD)
    return gradmag

##

from matplotlib.colors import LogNorm


for v in ds.data_vars:
    if not v.startswith('dist'): continue
    print(v)

    gradmag=dist_ratio(ds[v].values)
    
    fig=plt.figure(1)
    fig.clf()
    ax=fig.add_axes([0,0,1,1])
    cax=fig.add_axes([0.05,0.05,0.03,0.25])

    ccoll=g.plot_cells(values=gradmag,cmap='jet',norm=LogNorm(vmin=0.1,vmax=10),ax=ax)
    # ccoll.set_clim([0.1,10])
    plot_utils.cbar(ccoll,cax=cax)
    cax.set_title(r'$|| \nabla D ||$')
    ax.axis('equal')
    ax.axis( (601757, 628944., 4220689, 4249561) )
    ax.text(0.05,0.9,v,transform=ax.transAxes)

    fig.savefig('gradient_%s.png'%v,dpi=200)
