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

from .lego5 import Topography

@jit
def triangle_height(p1, p2, p3, a, b):
    """
    Input: 3 3-d (or higher) vectors p1, p2, p3
           2 3-d work arrays a, b
    Returns perpendicular distance of 3-vector p2 from line between p1 & p3
    |x,y,z| is magnitude of x-product of vectors p2-p1 (=a) & p3-p1 (=b),
    = 2 x area of triangle p1, p2, p3
    = area of rectangle length |p3-p1| x perp distance to p2
    Divide by distance |p3-p1| to get perp distance of p2
    """
    # a, b = p2 - p1, p3 - p1
    for i in range(3):
        a[i], b[i] = p2[i] - p1[i], p3[i] - p1[i]

    # (x,y,z) = a x b
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]

    # return |a x b|/|b|
    return np.sqrt((x*x + y*y +z*z)/(1.e-16 + b[0]*b[0] + b[1]*b[1] + b[2]*b[2]))


@jit
def quick_pass(keep, pts, n, work1, work2, threshold):
    """
    Input: Boolean array keep length n set to True
           n x 3+  array pts
           2 x 3   work arrays work1, work2
           threshold float
    Sets   keep[i] = False
    where perp distance of pts[i] from line between pts[i+1] & pts[ilo] < threshold
    """
    ilo = 0
    for i in range(1,n-1):
        dev = triangle_height(pts[ilo,:],pts[i,:],pts[i+1,:], work1, work2)
        if dev < threshold:
            keep[i] = False
        else:
            ilo = i

def quick_simplify(threshold, pts, passes=1):
    """
    Input: nmax x 3+ array pts to be simplified
           remove pts < threshold away from line after removal.
           run through quick_pass passes times, incrementing threshold
           from threshold/passes, 2*threshold/passes, ....,threshold
    Returns n(<nmax) x 3 simplified array
    """
    nmax, ndims = pts.shape
    # create 2 3d work arrays
    work1, work2 = np.empty([2,3],dtype = pts.dtype)
    # & boolean array that will be used to indicate points to keep/discard
    keep = np.empty([nmax],dtype=np.bool)

    # length of array of 3+ vectors npts will change, but set to initial value
    n = nmax
    # loop over increasing thr = threshold/passes, 2*threshold/passes, ...
    for i,thr in enumerate(np.linspace(0.,threshold, passes+1)[1:]):
        # keep all points until rejected
        keep[:n] = True
        # do the pass & find values to reject as keep[i] = False
        quick_pass(keep[:n], pts, n, work1, work2, thr)
        # get new, shorter array pts by removing points  with keep[i] = False
        pts = np.compress(keep[:n], pts, axis=0)
        # reverse order of pts so pass back and forth for symmetry while still in loop
        # ... also may need to reverse to ensure final pts array retains original sense
        if i< passes-1 or passes % 2 == 0:
            pts[...] = pts[::-1,:]
        n, _ = pts.shape

    return pts

def dopict(traj):
    """
    Interior function that masks, projects, and simplifies lth trajectory in
    traj_lon, traj_lat, traj_depth, traj_time, ...
    Returns simplified trajectory as an npts x 4 array xyzs
    """
    # mask before projection so invalid values -999.999 remain
    keep = traj['lon']>-999.
    for var in traj.keys():
        traj[var] = np.compress(keep, traj[var])

    # only project x, y, z
    if GetTraj.proj:
         GetTraj.proj(traj['lon'], traj['lat'], traj['depth'])

    # new version slower for short trajectories for reasonable number of passes, but faster for long trajectories ...
    xyzs = quick_simplify(GetTraj.threshold, np.vstack([traj[var] for var in GetTraj.vars]).T.copy(),
                           passes=GetTraj.passes)
    #print('npts=', xyzs.shape[0], end = '  ')
    sys.stdout.flush()
    return xyzs


class GetTraj(object):
    @classmethod
    def setvars(cls, *vars):
        """
        Set optional extra traj_xxx data to read
        Always reads lat, lon, depth
        """
        GetTraj.vars = ('lon','lat','depth') + vars

    def __init__(self, pathname, icb=False):
        """
        Set name of trajectory file
        """
        self.pathname = pathname
        self.icb = icb

    def __getitem__(self,slice):
        """
        Reads trajectories from file self.pathname for variables in self.vars
        into dictionary self.traj['lon'], self.traj['lat'] ....
        """
        with Dataset(self.pathname) as f:
            fv = f.variables
            self.traj = {}
            if self.icb:
               for var in self.vars:
                   if var != 'depth' :
                    try:
                      self.traj[var] = fv['%s' % var][slice]
                    except:
                      print(var,' not found in dataset.')
                      print('If trajectories are not iceberg trajectories,')
                      print('re-run without the --icb flag\n')
                      sys.exit()
               self.traj['depth'] = 1.0 + 0.0*self.traj['lon']
            else:
               for var in self.vars:
                   try:
                     self.traj[var] = fv['traj_%s' % var][slice]
                   except:
                     print('traj_'+var,' not found in dataset.')
                     print('If trajectories are iceberg trajectories,')
                     print('re-run with the --icb flag\n')
                     sys.exit()

    def get_traj_list(self, killdict=True):
        """
        Create list of trajectories [[traj_1[lon], traj_1[lat], ...], [traj_2[lon],...], ....
        """
        if self.icb:
           ntraj, ltraj = self.traj['lon'].shape
           self.traj_list  = []
           for l in range(ntraj):
               self.traj_list.append({var:x[l,:] for var,x in
                                      [(var,self.traj[var]) for var in GetTraj.vars]})
        else:
           ltraj, ntraj = self.traj['lon'].shape
           self.traj_list  = []
           for l in range(ntraj):
               self.traj_list.append({var:x[:,l] for var,x in
                                      [(var,self.traj[var]) for var in GetTraj.vars]})
        if killdict: del self.traj

    def prepare_trajectories(self, proj = None, threshold_deg = 0.01, passes = 50):
        """
        Masks, projects, and simplifies trajectories in self.traj_list
        Returns simplified trajectories as a list of npts x nvar arrays xyzs_list
        """
        GetTraj.proj = proj
        # should be threshold_deg latitude
        GetTraj.threshold = 6e6*(np.pi/180.)*threshold_deg
        GetTraj.passes = passes

        # loop over trajectories
        # set # of processes for parallel processing
        processes = max(cpu_count() - 2,1)
        #processes = 1
        if processes > 1 :
           print('trajectory processing will use ',processes,' threads\n')
        sys.stdout.flush()
        if processes>1:
            # create processes workers
            pool = Pool(processes=processes)
            # append output from dopict to xyzs_list for l=0, 1, ...ntraj-1
            xyzs_list = pool.map(dopict,self.traj_list)
            # wait for all workers to finish & exit parallellized stuff
            pool.close()
        else:
            xyzs_list = []
            for traj in self.traj_list:
                xyzs_list.append( dopict(traj) )
        del self.traj_list
        return xyzs_list


def do_trajectories(traj, traj_numbers, topo, icb,
                    xs=None, xe=None, ys=None, yn=None,
                    traj_cut=None,
                    threshold_deg=0.01, passes=50):
    t1 = time.time()
    nt0, nt1, dnt = traj_numbers
    if icb:
       GetTraj.setvars('ttim')
       gt = GetTraj(traj, icb=icb)
       gt[nt0:nt1:dnt, :traj_cut]
    else:
       GetTraj.setvars('time')
       gt = GetTraj(traj, icb=icb)
       gt[:traj_cut, nt0:nt1:dnt]
    t1, t0 = time.time(), t1
    print('%10.5f s taken to read trajectories\n' % (t1 - t0) )

    gt.get_traj_list()
    xyzs_list = gt.prepare_trajectories(proj=topo.proj, threshold_deg=threshold_deg,
                                         passes=passes)
    endpts = []
    endpts.append(0)
    rlen = 0
    ntrj = 0
    for xyzs in xyzs_list:
        rlen += xyzs.shape[0]
        ntrj = ntrj + 1
        endpts.append(rlen)
    xxtrac, yytrac, zztrac, sstrac  = np.concatenate(xyzs_list).T.copy()
    del xyzs_list
    t1, t0 = time.time(), t1
    print('%10.5f s taken to prepare ' % (t1 - t0),ntrj,' trajectories\n')

    #print('\n', len(endpts), endpts)
    connections = []
    ept = 0
    for i in range(1,len(endpts)):
        connections.append(np.c_[np.arange(ept, endpts[i] -2),
                                np.arange(ept+1,endpts[i] -1 )])
        ept = endpts[i]

    del endpts
    t1, t0 = time.time(), t1
    print('%10.5f s taken to connect trajectories\n' % (t1 - t0) )

    pts = mlab.pipeline.scalar_scatter(xxtrac,yytrac,zztrac,sstrac)
    pts.mlab_source.dataset.lines = np.vstack(connections)

    lines = mlab.pipeline.surface(
                    mlab.pipeline.tube(
                      mlab.pipeline.stripper(
                            pts,
                        ),
                          tube_sides=7, tube_radius=topo.zscale*20,
                    ),
                    colormap='Spectral',
                )

    # A view of the canyon
    t1, t0 = time.time(), t1
    print('%10.5f s taken to draw trajectories\n' % (t1 - t0) )
    if platform.system() == "Linux":
      # Linux systems return memory in Kbytes
      print('peak memory usage is (MB):',
            ' self:',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            ' children:',resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024
            ,'\n')
    else:
      # Assumed MACOS type (Darwin) return in bytes
      print('peak memory usage is (MB):',
            ' self:',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024),
            ' children:',resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/(1024*1024)
            ,'\n')
    #mlab.show()

if __name__ == '__main__':
    parser = ArgumentParser(description='produce lego-block topography e.g. \n python ~/VC_code/NEMOcode/lego5.py -b  0 10000 600 10000 -d ../025')
    parser.add_argument('-b',dest='bounds',help='ilo ihi jlo jhi', type=int, nargs= '*',default=None)
    parser.add_argument('--ilo',dest='ilo',help='ilo; overrrides value in bounds', type=int, default=None)
    parser.add_argument('--jlo',dest='jlo',help='jlo; overrrides value in bounds', type=int, default=None)
    parser.add_argument('--ihi',dest='ihi',help='ihi; overrrides value in bounds', type=int, default=None)
    parser.add_argument('--jhi',dest='jhi',help='jhi; overrrides value in bounds', type=int, default=None)

    parser.add_argument('--bathy',dest='bathymetry_file',help='bathymetry file if not bathy_meter.nc',
                        default='bathy_meter.nc')
    parser.add_argument('--coords',dest='coordinate_file',help='coordinate file if not coordinates.nc or mesh_hgr.nc',
                        default='coordinates.nc')
    parser.add_argument('--domain','-d',dest='domain_dir',help='directory of bathymetry & coordinates',
                        default='./')
    parser.add_argument('--bottom',dest='bottom',type=float,
                         help='(positive) depth where colorscale saturates to deepest value',
                        default=6000.)
    parser.add_argument('--globe','-g', dest='globe',action='store_true',
                         help='do globe', default=False)
    parser.add_argument('--traj','-t', dest='traj', help='file with trajectories', default=None)
    parser.add_argument('--traj_numbers',dest='traj_numbers', type=int, nargs= 3, help='trajectory_numbers', default=[None, None, None])
    parser.add_argument('--traj_cut',dest='traj_cut', type=int, help='last trajectory time level to read', default=None)
    parser.add_argument('--traj_threshold',dest='threshold', type=float,help='threshold in deg for shortening trajectories', default=0.01)
    parser.add_argument('--passes',dest='passes', help='# of passes for path simplification using quick_simplify',
                        type=int, default=50)
    args = parser.parse_args()

    if args.bounds is None:
        xs, xe = None, None
        ys, yn = None, None
    else:
        xs, xe = args.bounds[:2]
        ys, yn = args.bounds[2:]

    if args.ilo is not None:
        xs = args.ilo
    if args.jlo is not None:
        ys = args.jlo
    if args.ihi is not None:
        xe = args.ihi
    if args.jhi is not None:
        yn = args.jhi

    if args.globe:
        map = None
    else:
        # Use a basemap projection; see http://matplotlib.org/basemap/users/mapsetup.html
        # Lambert conformal
        # m = Basemap(llcrnrlon=-95.,llcrnrlat=1.,urcrnrlon=80.,urcrnrlat=80.,\
        #         rsphere=(6378137.00,6356752.3142),\
        #         resolution='l',area_thresh=1000.,projection='lcc',\
        #         lat_1=50.,lon_0=-35.)
        # Orthographic (still won't work)
        # map = Basemap(projection='ortho',lat_0=50.,lon_0=-35.)
        # Mollweide
        # map = Basemap(projection='moll',lon_0=0,resolution='c')
        # N Polar stereographic
        map = Basemap(projection='npstere',boundinglat=10,lon_0=270,resolution='l')


    topo = Topography(xs=xs, xe=xe, ys=ys, yn=yn,
                     domain_dir=args.domain_dir, bathymetry_file=args.bathymetry_file,
                     coordinate_file= args.coordinate_file,
                     bottom = args.bottom, map = map, globe = args.globe)

    if args.traj is not None:
        do_trajectories(args.traj, args.traj_numbers, topo,
                        xs=xs, xe=xe, ys=ys, yn=yn,
                        traj_cut=args.traj_cut,
                        threshold_deg=args.threshold, passes=args.passes)

