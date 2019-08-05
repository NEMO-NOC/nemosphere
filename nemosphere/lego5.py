#!/usr/bin/env python
from __future__ import print_function
import os, sys, platform
from argparse import ArgumentParser
import numpy as np
import time
import resource

from mayavi import mlab
from netCDF4 import Dataset

from mpl_toolkits.basemap import Basemap

from numba import jit

@jit
def add_triangles_from_square(x, x1, x2, x3, x4, k):
    '''
    inserts values of kth and k+1th triangles into array x in place
    from face values x1, x2, x3, x4
    '''
    k1 = k + 1
    x[k,0], x[k,1], x[k,2] = x1, x2, x3
    x[k1,0], x[k1,1], x[k1,2] = x2, x3, x4


@jit
def get_triangles_j(j, nx, k, t_land, ds_max, lambda_f, phi_f, dep,
                     x, y, z, t):
    '''
    inserts coordinates of & col shading for triangles around ij points on j-line:
     -- horz face centred on T points at dep[j,i]
     + surrounding vertical faces (only count where neighvouring point is shallower
        to avoid double counting)
    into max_trianglesx3 arrays x, y, z, t,
    starting with the horz face at
    x[k,1..3], y[k,1..3], z[k,1..3] and t[k,1..3]

    On land points, with dep[j,i]==0 t[k,1..3] set to t_land
    '''
    jm1, jp1 = j-1, j+1
    xx01, xx11, yy01, yy11 = lambda_f[jm1,0], lambda_f[j,0], phi_f[jm1,0],phi_f[j,0]
    for i in range(1, nx-1):
        im1, ip1 = i-1, i+1
        xx00, xx10, yy00, yy10 = xx01, xx11, yy01, yy11
        xx01, xx11, yy01, yy11 = lambda_f[jm1,i], lambda_f[j,i], phi_f[jm1,i],phi_f[j,i]
        if abs(xx01 - xx00) + abs(yy01 - yy00) + abs(xx11 - xx10) + abs(yy11 - yy10) > ds_max:
            continue

        # x & y coordinates of f-points surrounding T-point i,j
        # 00 = SW, 10 = NW, 01 = SE, 11 = NE

        # do horizontal faces of T-box, zig-zag points SW, NW, SE, NE
        # insert x & y for 2-triangles (kth & k+1th)
        add_triangles_from_square(x, xx00, xx10, xx01, xx11, k)
        add_triangles_from_square(y, yy00, yy10, yy01, yy11, k)
        # .. constant z
        dep00 = dep[j,i]
        add_triangles_from_square(z, dep00, dep00, dep00, dep00, k)
        # color depends on z
        if dep00 == 0.:
            add_triangles_from_square(t, t_land, t_land, t_land, t_land, k)
        else:
            add_triangles_from_square(t, dep00, dep00, dep00, dep00, k)
        # & increment k by 2
        k += 2

        # do vertical faces surrounding T-box
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            dep01 = dep[j+dj, i+di]
            if dep01 > dep00:
                #  vertical face zig-zag z points:
                add_triangles_from_square(z, dep00, dep01, dep00, dep01, k)
                # color is shaded
                add_triangles_from_square(t, dep00, dep01, dep00, dep01, k)
                if di==-1:
                    # face at u-points, constant i
                    add_triangles_from_square(x, xx00, xx00, xx10, xx10, k)
                    add_triangles_from_square(y, yy00, yy00, yy10, yy10, k)
                elif di==1:
                    add_triangles_from_square(x, xx01, xx01, xx11, xx11, k)
                    add_triangles_from_square(y, yy01, yy01, yy11, yy11, k)
                elif dj ==-1:
                    # face at v-points, constant j
                    add_triangles_from_square(y, yy00, yy00, yy01, yy01, k)
                    add_triangles_from_square(x, xx00, xx00, xx01, xx01, k)
                elif dj ==1:
                    add_triangles_from_square(y, yy10, yy10, yy11, yy11, k)
                    add_triangles_from_square(x, xx10, xx10, xx11, xx11, k)
                k += 2
    return k


def get_triangles(dep, lambda_f, phi_f, t_land, ds_max):
    '''
    takes 2-D array of depths dep, assumed to be positioned at j & i values
    Creates mesh of triangles covering lego-block topography consistent with dep
    Outputs four ntriangles x 3 arrays x, y, z, t where
    x(k,1..3), y(k,1..3), z(k,1..3) and t(k,1..3) are the x, y, z and color values for the kth triangle
    '''
    # arrays in C-order so last index is x
    ny,nx = dep.shape
    # max_no_triangles is maximum number of triangles ....
    #  (ny-2)*(nx-2) is npts with all 4 sides available
    #  factor of 3 for top and 2 sides; factor of 2 since 2 triangles in each face
    #  add 2*(ny-2+nx-2) since edge interfaces not accounted for
    max_no_triangles = (ny-2)*(nx-2)*3*2 + 2*(ny-2+nx-2)

    # can iterate through 1st '0th' index of array, to give 4 2d arrays max_triangles x 3
    x, y, z, t = np.zeros((4, max_no_triangles, 3), dtype=dep.dtype)

    # first array created will be for first (0th) triangle
    k = 0

    # loop through each j-line of T-points ...
    # note range(m,n) = (m, ....n-1)
    for j in range(1, ny-1):
        # get triangles for all i-points on j-line
        k = get_triangles_j(j, nx, k, t_land, ds_max, lambda_f, phi_f, dep, x, y, z, t)

    # k is now total no of triangles; chop off unused parts of the arrays & copy ...
    x, y, z, t = [a[:k,:].copy() for a in (x, y, z, t)]
    return k, x, y, z, t

def wrap_lon(lon):
    """
    Ensures longitude is between -180 & 180. Not really necessary.
    """
    # Need [] to ensure lon is changed in-place instead of making new variable
    lon[...] = (lon[...] + 180.) % 360. - 180.

def find_domain_file(domain_dir, file_list):
    for filename in file_list:
        pathname = os.path.join(domain_dir,filename)
        if os.path.exists(pathname):
            return pathname
    else:
        sys.exit('cannot find any of %s in %s' % (' '.join(file_list), domain_dir ))


class Topography(object):
    def __init__(self, xs=None, xe=None, ys=None, ye=None,
                     domain_dir='.', bathymetry_file='bathy_meter.nc', coordinate_file='coordinates.nc',
                     bottom = 6000., cmap='gist_earth', topo_cbar=False, map2d = None, globe = False, zs_rat = 0.1,
                     size_in_pixels = (1000,800)):
        # xem1, yem1 = xe - 1, ye - 1
        xem1, yem1 = xe, ye

        t1 = time.time()
        pathname = find_domain_file(domain_dir,[bathymetry_file, 'allmeshes.nc'])
        with Dataset(pathname) as f:
            # print(f.variables.keys())
            dep = f.variables['Bathymetry'][ys:ye,xs:xe].astype(np.float32)

        pathname = find_domain_file(domain_dir,['mesh_hgr.nc', 'allmeshes.nc', coordinate_file])
        with Dataset(pathname) as f:
            # print(f.variables.keys())
            lambda_f = f.variables['glamf'][...,ys:ye,xs:xe].squeeze().astype(np.float32)
            phi_f = f.variables['gphif'][...,ys:ye,xs:xe].squeeze().astype(np.float32)
        t1, t0 = time.time(), t1
        print('%10.5f s taken to read in data\n' % (t1 - t0) )

        if globe:
            # Plug the South Pole if the bathymetry doesn't extend far enough
            minlat = phi_f[:,0].min()
            if minlat > -89.9 and minlat < -75.:
                nj,ni = phi_f.shape
                nextra = 10
                dy_deg =  (minlat + 90.)/nextra
                lonfill = np.empty((nj+nextra,ni), dtype=lambda_f.dtype)
                latfill = np.empty((nj+nextra,ni), dtype=phi_f.dtype)
                depfill = np.empty((nj+nextra,ni), dtype=dep.dtype)
                lonfill[nextra:,:] = lambda_f
                latfill[nextra:,:] = phi_f
                depfill[nextra:,:] = dep
                lonfill[:nextra,:] = lambda_f[0,:]
                # Add new dimension None to 1D y-array so it can be 'Broadcast' over longitude
                latfill[:nextra,:] = np.arange(-90,minlat,dy_deg, dtype=np.float32)[:,None]
                depfill[:nextra,:] = 0.0
                phi_f, lambda_f, dep = latfill, lonfill, depfill
                del latfill, lonfill, depfill
            # Ellipsoidal earth
            self.rsphere_eq, self.rsphere_pol = 6378137.00, 6356752.3142
            dist =  self.rsphere_eq + self.rsphere_pol
            self.proj = self.globe_proj

        elif map2d is not None:
            wrap_lon(lambda_f)
            lambda_f, phi_f = map2d(lambda_f, phi_f)

            # need to scale heights/depths for consistency with picture using horizontal axes i & j
            dlam = lambda_f.max() - lambda_f.min()
            dphi = phi_f.max() - phi_f.min()
            dist = np.sqrt(dlam*dlam + dphi*dphi)
            self.map2d = map2d
            self.proj = self.map_proj


        ny, nx = lambda_f.shape
        ds_max = 20.*dist/max(ny,nx)

        #   ... and convert from depths--> heights
        #   ... and scale depth of saturated colorscale

        zscale = zs_rat*dist/6000.
        self.zscale = zscale
        dep = -zscale*dep #.astype(np.float64)

        t1, t0 = time.time(), t1
        print('%10.5f s taken to scale & convert data to float64\n' % (t1 - t0) )

        # colors of flat steps are associated with their depth
        # reset color for land points to positive value, so land in uppermost color class
        # Since there are 256 color classes, only want land in uppermost 1/256 of color values.
        # i.e. If dt = (t_land - t_min), need t_land such that t_land - dt/256 =0,
        # so > all ocean colour values t<0
        zs_min = -zscale*bottom
        t_land = np.NaN
        ntriangles, x, y, z, t = get_triangles(dep, lambda_f, phi_f, t_land, ds_max)
        # ntriangles is now total no of triangles; flatten to 1d arrays of vertices of length 3*triangles
        x, y, z, t = [a.ravel() for a in (x, y, z, t)]
        # where triangles(k,1...3) contains indices relating to kth triangle
        triangles = np.arange(3*ntriangles,dtype=np.int32).reshape(ntriangles,3)
        t1, t0 = time.time(), t1
        print('%10.5f s taken to calculate vertices\n' % (t1 - t0) )

        if globe:
            z /= zscale
            self.proj(x,y,z)
            t1, t0 = time.time(), t1
            print('%10.5f s taken to transform triangles onto sphere\n' % (t1 - t0) )

        #clear some memory; mayavi consumes a lot;)
        del dep

        self.cmap = cmap
        self.vmin = zs_min
        fig = mlab.figure(size=size_in_pixels, bgcolor=(0.16, 0.28, 0.46))
        columns = mlab.triangular_mesh(x, y, z, triangles, colormap=cmap, scalars=t, vmin=zs_min)

        # Set color for NAN's (e.g. land)
        # columns representing RGBA (red, green, blue, alpha) coded with floats from 0. to 1.
        # Black
        NaN_color = (0.,0.,0.,1.0)
        # dark red/brown
        # NoN_color = (0.3,0.05,0.05,1.0)
        columns.module_manager.scalar_lut_manager.lut.nan_color = NaN_color

        if topo_cbar:
            xd, yd, zd = [coordinate[:1].copy() for coordinate in (x,y,z)]
            sd = [0.]
            dummy = mlab.points3d(xd, yd, zd, sd, scale_factor=0.0001,
                                   colormap = self.cmap, vmin=-6000., vmax=-0)
            dummy.glyph.color_mode = 'color_by_scalar'
            dummy.glyph.glyph_source.glyph_source.center = [0, 0, 0]
            cbar = mlab.scalarbar(object=dummy, title='Bottom Depth', nb_labels=6, label_fmt='%4.0f')
            cbar.scalar_bar_representation.maximum_size = np.array([50000, 50000])
            cbar.scalar_bar_representation.position = [0.3, 0.15]
            cbar.scalar_bar_representation.position2 = [0.4, 0.05]
        t1, t0 = time.time(), t1
        print('%10.5f s taken to setup topography plot\n' % (t1 - t0) )


        # need handle to figure & mapping transformation
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
        # self.columns = columns

    def map_proj(self, x, y, z):
        x[...],y[...] = self.map2d(x, y)
        z[...] *= self.zscale

    def globe_proj(self, x, y, z):
        r_eq, dr_pol_eq = self.rsphere_eq, self.rsphere_pol - self.rsphere_eq
        rad = np.pi/180.
        z[...] = z*self.zscale + r_eq
        y *= rad
        x *= rad
        xt = z*np.cos(y)
        z[...]  = (z + dr_pol_eq)*np.sin(y)
        y[...] = xt*np.sin(x)
        x[...] = xt*np.cos(x)


if __name__ == '__main__':
    parser = ArgumentParser(description='produce lego-block topography e.g. \n python ~/VC_code/NEMOcode/lego5.py -b  0 10000 600 10000 -d ../025')
    parser.add_argument('-b',dest='bounds',help='ilo (f or u) ihi jlo (f or v) jhi', type=int,
                         nargs= '*',default=None)
    parser.add_argument('--size',dest='size_in_pixels',help='nx ny', type=int,
                         nargs= '*',default=(1024, 768))
    parser.add_argument('--ilo',dest='ilo',help='ilo; overrrides value in bounds', type=int, default=None)
    parser.add_argument('--jlo',dest='jlo',help='jlo of southern f (or v) -point bdry', type=int, default=None)
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
    parser.add_argument('--topo_cbar', dest='topo_cbar',action='store_true',
                         help='draw topography colorbar', default=False)
    args = parser.parse_args()

    if args.bounds is None:
        xs, xe = None, None
        ys, ye = None, None
    else:
        xs, xe = args.bounds[:2]
        ys, ye = args.bounds[2:]

    if args.ilo is not None:
        xs = args.ilo
    if args.jlo is not None:
        ys = args.jlo
    if args.ihi is not None:
        xe = args.ihi
    if args.jhi is not None:
        ye = args.jhi

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


    topo = Topography(xs=xs, xe=xe, ys=ys, ye=ye,
                        domain_dir=args.domain_dir, bathymetry_file=args.bathymetry_file,
                        coordinate_file= args.coordinate_file, size_in_pixels = args.size_in_pixels,
                        bottom = args.bottom, map2d = map, globe = args.globe, topo_cbar = args.topo_cbar)


    mlab.show()
