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
g=unstructured_grid.UnstructuredGrid(max_sides=4)
g.add_rectilinear([0,0],[1000,500],41,21)
g.write_ugrid('rectangle.nc')
##

g=unstructured_grid.UnstructuredGrid.from_ugrid('rectangle-branch_and_wiggle.nc')

## 
plt.figure(1).clf()
g.plot_edges()
plt.axis('equal')

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

def fig_dist(C,num=2,log=False,title="",local_max=False, direction=False,
             nbrhood=4):
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
        idxs=np.arange(g.Ncells())
        np.random.shuffle(idxs)
        samp_cells=idxs[:1000]
        cc=g.cells_center()
        XY=cc[samp_cells]
        UV=np.nan*XY
        for i,c in utils.progress(enumerate(samp_cells)):
            cells=[c]
            for _ in range(nbrhood): cells=np.unique(list(cells) + [c for c0 in cells for c in g.cell_to_cells(c0) if c>=0] )
            #y=cell_costs[cells]
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


##

ds=g.write_to_xarray()

x_down=np.array([10,10])

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

fig=fig_dist(cell_costs,num=6,log=False,title='Dijkstra',local_max=True,
             direction=True,nbrhood=2)


ds['dist_dijk']=('face',),cell_costs
ds['dist_dijk'].attrs['desc']="Dijkstra"

##

gd=unstructured_diffuser.Diffuser(g)
gd.set_flux(xy=x_down, value=10.0)
gd.set_decay_rate(1e-8)
gd.construct_linear_system()
C=gd.compute()
D=cost_to_dist(-C)
fig=fig_dist(D,num=2,log=False,title="Diffusion and decay, no bathy",local_max=False,
             direction=True,nbrhood=2)

ds['dist_diff']=('face',),D
ds['dist_diff'].attrs['desc']="K=100, decay=1e-8, downstream flux=10"
