#!/usr/bin/env python
from __future__ import print_function
import os, sys, platform
from argparse import ArgumentParser
import numpy as np
import numpy.ma as ma
import time
import resource

from mayavi import mlab
from netCDF4 import Dataset

from numba import jit
from multiprocessing import Pool,cpu_count

from mpl_toolkits.basemap import Basemap

def dopict(traj):
    """
    Interior function that masks, projects, and simplifies lth trajectory in
    traj_lon, traj_lat, traj_depth, traj_time, ...
    Returns simplified trajectory as an npts x 4 array xyzs
    """

    # only project x, y, z
    if GetTraj.proj:
         GetTraj.proj(traj['lon'], traj['lat'], traj['depth'])

    # new version slower for short trajectories for reasonable number of passes, but faster for long trajectories ...
    xyz = (traj['lon'], traj['lat'], traj['depth'])
    # sys.stdout.flush()
    return xyz


class GetTraj(object):
    @classmethod
    def setvars(cls, *vars):
        """
        Set optional extra traj_xxx data to read
        Always reads lat, lon, depth
        """
        GetTraj.vars = ('lon','lat','depth') + vars

    def __init__(self, pathname, proj = None):
        """
        Set name of trajectory file & projection
        """
        self.pathname = pathname
        GetTraj.proj = proj

    def __getitem__(self,slice):
        """
        Reads trajectories from file self.pathname for variables in self.vars
        into dictionary self.traj['lon'], self.traj['lat'] ....
        """
        with Dataset(self.pathname) as f:
            fv = f.variables
            self.traj = {}
            for var in self.vars:
                self.traj[var] = fv['traj_%s' % var][slice].compressed()

    def get_variable(self,var, nt, stride=1):
        """
        Reads trajectories from file self.pathname for variables in self.vars
        into dictionary self.traj['lon'], self.traj['lat'] ....
        """
        with Dataset(self.pathname) as f:
            fv = f.variables
            return fv['traj_%s' % var][nt,::stride].compressed()

    def prepare_dots(self, nt, stride=1):
        """
        Masks, projects, and simplifies trajectories in self.traj_list
        Returns simplified trajectories as a list of npts x nvar arrays xyzs_list
        """
        self[nt,::stride]

        # loop over trajectories
        # set # of processes for parallel processing
        processes = max(cpu_count() - 2,1)
        processes = 1
        if processes > 1 :
           print('trajectory processing will use ',processes,' threads\n')
        sys.stdout.flush()
        if processes>1:
            # create processes workers
            pool = Pool(processes=processes)
            # append output from dopict to xyzs_list for l=0, 1, ...ntraj-1
            xyz = pool.map(dopict,self.traj)
            # wait for all workers to finish & exit parallellized stuff
            pool.close()
        else:
            xyz = dopict(self.traj)
        #del self.traj
        return xyz



def do_dots(traj, ntimes, topo, stride=40):
    t1 = time.time()
    if len(ntimes)==3:
        nt0, nt1, dnt = ntimes
        ntimes = range(nt0, nt1, dnt)

    GetTraj.setvars('time')
    gt = GetTraj(traj, topo.proj)
    t1, t0 = time.time(), t1
    print('%10.5f s taken to read trajectories\n' % (t1 - t0) )

    flat = False
    for nt in ntimes:
        print('doing time level', nt)
        x, y, z = gt.prepare_dots(nt, stride=stride)
        s = np.zeros_like(x) + 1.
        zreal = gt.get_variable('depth', nt, stride=stride)
        if flat:
            pts = mlab.points3d(x, y, z, s, scale_factor=topo.zscale*100, color=(1,1,1), opacity=0.7)
        else:
            n = len(x)
            pts = mlab.quiver3d(x, y, z, s, s, s, scalars=zreal, mode='sphere', scale_factor=topo.zscale*100, vmin=-3000., vmax=-500., opacity=1.)#colormap = topo.cmap, vmin = topo.vmin, vmax=0.)
            pts.glyph.color_mode = 'color_by_scalar'
            pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]

            cbar = mlab.scalarbar(object=pts, title='Depth', nb_labels=6, label_fmt='%4.0f')
            cbar.scalar_bar_representation.maximum_size = np.array([50000, 50000])
            cbar.scalar_bar_representation.position = [0.3, 0.15]
            cbar.scalar_bar_representation.position2 = [0.4, 0.05]
    t1, t0 = time.time(), t1
    print('%10.5f s taken to draw trajectories\n' % (t1 - t0) )
