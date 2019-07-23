import os
from stompy.model.fish_ptm import ptm_tools
from stompy.model.suntans import sun_driver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

plt.figure(1)
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
df['x0']=labels[parts['id']]
df['xN']=parts['x'][:,0]

# bin particles by starting label

grps=df.groupby(np.round(df.x0/100))

grouped=pd.DataFrame( dict(x0_mean=grps['x0'].mean(),
                           x0_var=grps['x0'].var(),
                           xN_var=grps['xN'].var(),
                           xN_size=grps.size() ))

##  how many particles started in that group?
orig=pd.Series(parts0['x'][:,0],name='x')
x0_size=orig.groupby(np.round(orig/100)).size()
grouped['x0_size']=x0_size.loc[grouped.index]

##

grouped['K']=(grouped['xN_var']-grouped['x0_var'])/hydro_period

fig=plt.figure(3)
fig.clf()
ax1=fig.add_subplot(1,1,1)
ax1.plot(grouped.x0_mean,grouped.K,label='K',color='g')
ax_count=ax1.twinx()
ax_count.plot(grouped.x0_mean,grouped.x0_size-grouped.xN_size,
              label='loss',color='0.6')
ax1.set_ylabel('K ($m^2 s^{-1}$)',color='g')
ax_count.set_ylabel('Particles lost',color='0.6')
plt.setp(ax_count.get_yticklabels(),color='0.6')
plt.setp(ax1.get_yticklabels(),color='g')
ax1.spines['left']
ax1.spines['left'].set_color('g')
ax1.spines['right'].set_visible(0)
ax1.spines['top'].set_visible(0)
ax_count.spines['left'].set_visible(0)
ax_count.spines['right'].set_color('0.6')
ax_count.spines['top'].set_visible(0)
ax1.set_xlabel("Distance from tidal BC (m)")
fig.savefig("channel-dispersion-v0.png")

