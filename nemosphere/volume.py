from __future__ import print_function
import sys, os
import numpy as np
import numpy.ma as ma
from os.path import join as pjoin
import os.path

from numba import jit
from multiprocessing import Pool,cpu_count

from skimage.measure import marching_cubes
from netCDF4 import Dataset
import time
import warnings

from mayavi import mlab

from lego5 import find_domain_file


@jit('f8(f8, f8, f8, f8, f8, f8, f8, f8, f8)',nopython=True)
def triangle_length(x0, x1, x2, y0, y1, y2, z0, z1, z2):
    """
    Input: 3 3-d (or higher) vectors p1, p2, p3
    Returns perpendicular distance of 3-vector p2 from line between p1 & p3
    |x,y,z| is magnitude of x-product of vectors p2-p1 (=a) & p3-p1 (=b),
    = 2 x area of triangle p1, p2, p3
    = area of rectangle length |p3-p1| x perp distance to p2
    Divide by distance |p3-p1| to get perp distance of p2
    """
    length2 = 0.333*(
        (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
        + (x0-x2)*(x0-x2) + (y0-y2)*(y0-y2) + (z0-z2)*(z0-z2)
        + (x0-x1)*(x0-x1) + (x0-x1)*(x0-x1) + (z0-z1)*(z0-z1)
         )
    return length2

@jit('void(i4, i8[:,:], f8[:], f8[:], f8[:], b1[:], f8)',nopython=True)
def find_long_faces(nfaces, faces, x, y, z, keep_face, too_large_deg2):
    for i in range(nfaces):
        nf0, nf1, nf2 = faces[i,0], faces[i,1], faces[i,2]
        l2 = triangle_length( x[nf0], x[nf1], x[nf2], y[nf0], y[nf1], y[nf2],
                              z[nf0], z[nf1], z[nf2] )
        if l2 > too_large_deg2:
            keep_face[i] = False

def remove_long_faces(faces, x, y, z, too_large_deg):
    too_large_deg2 = too_large_deg**2
    nfaces, _ = faces.shape
    keep_face = np.ones([nfaces],dtype=np.bool)
    find_long_faces(nfaces, faces, x, y, z, keep_face, too_large_deg2)
    faces = np.compress(keep_face,faces, axis=0)
    return faces


@jit('void(i4, i8[:,:], f8[:,:], b1[:])',nopython=True)
def find_nan_faces(nfaces, faces, verts, keep_face):
    for i in range(nfaces):
        nf0, nf1, nf2 = faces[i,0], faces[i,1], faces[i,2]
        for nf in (nf0, nf1, nf2):
            if np.isnan(verts[nf,0]) or np.isnan(verts[nf,1]) or np.isnan(verts[nf,2]):
                keep_face[i] = False
                break

def remove_nan_faces(faces, verts):
    nfaces, _ = faces.shape
    keep_face = np.ones([nfaces],dtype=np.bool)
    find_nan_faces(nfaces, faces, verts, keep_face)
    faces = np.compress(keep_face,faces, axis=0)
    return faces

@jit('void(f8[:,:], i4, f8[:,:], f8[:,:], f8[:,:,:], i8[:,:], f8, f8)',nopython=True)
def do_ijk_to_lat_lon_height(v, nv, lat, lon, zt, kmt, rnx, rny):

    epsilon = 1.e-10

    for n in range(nv):
        if np.isnan(v[n,0])  or np.isnan(v[n,1]) or np.isnan(v[n,2]):
            v[n,0], v[n,1], v[n,2] = np.NaN, np.NaN, np.NaN
        else:
            rk, rj, ri = v[n,0] - epsilon, v[n,1] - epsilon, v[n,2] - epsilon
            k, j, i = int(rk), int(rj), int(ri)
            #if ri> rnx-2. or rj > rny-2.:
            if ri>= rnx-1. or rj >= rny-1.:
                v[n,0], v[n,1], v[n,2] = np.NaN, np.NaN, np.NaN
            elif rk > float(min(kmt[j,i], kmt[j+1,i+1], kmt[j+1,i], kmt[j,i+1])-1):
                #keep[n] = False
                v[n,0], v[n,1], v[n,2] = np.NaN, np.NaN, np.NaN
            else:
                dk, dj, di = rk - float(k), rj - float(j), ri - float(i)
                v[n,0] = zt[k, j, i]*(1. - dk) + zt[k+1, j, i]*dk
                v[n,1] = lat[j,i]*(1. - dj)*(1. - di) + lat[j+1,i]*dj*(1. - di) + \
                    lat[j,i+1]*(1. - dj)*di + lat[j+1,i+1]*dj*di
                v[n,2] = lon[j,i]*(1. - dj)*(1. - di) + lon[j+1,i]*dj*(1. - di) + \
                    lon[j,i+1]*(1. - dj)*di + lon[j+1,i+1]*dj*di

def ijk_to_lat_lon_height(v, lat, lon, zt, kmt):
    ny, nx = kmt.shape
    rny, rnx = float(ny), float(nx)
    nv, _ = v.shape
    do_ijk_to_lat_lon_height(v, nv, lat, lon, zt, kmt, rnx, rny)

class Surface():
    pass

def do_surface(value):
    t1 = time.time()
    # get triangles for surface using marching_cubes
    # suppressing warnings for NaNs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        verts, faces = marching_cubes(Surface.T, value)

    # adjust i and j components of vertices, so they match with lon & lat
    verts[:,0] += Surface.di
    verts[:,1] += Surface.dj
    print('max faces', faces.max(), 'faces shape', faces.shape, 'no verts =', verts.shape[0])
    print (verts.shape)
    t1, t0 = time.time(), t1
    print('time taken to calculate surface is', t1 - t0, ' s','\n')

    ijk_to_lat_lon_height(verts, Surface.lat, Surface.lon, Surface.height, Surface.kmt)
    t1, t0 = time.time(), t1
    print('time taken to go from ijk to latlondepth is', t1 - t0, ' s','\n')

    faces = remove_nan_faces(faces,verts)
    t1, t0 = time.time(), t1
    print('time taken to remove NaNs is', t1 - t0, ' s','\n')

    z, y, x = verts.T.copy()
    del verts
    Surface.proj(x,y,z)
    t1, t0 = time.time(), t1
    print('max faces', faces.max(), 'faces shape', faces.shape, 'no verts =', x.shape[0])
    print('time taken to project vertices is', t1 - t0, ' s','\n')

    faces = remove_long_faces(faces, x, y, z, Surface.too_large)
    t1, t0 = time.time(), t1
    print('max faces', faces.max(), 'faces shape', faces.shape, 'no verts =', x.shape[0])
    print('time taken to remove long triangles is', t1 - t0, ' s','\n')

    return x,y,z, faces

def do_vol(vble, fname, values, proj,
           xs=None, xe=None, ys=None, ye=None, domain_dir='.',
           coordinate_file = 'coordinates.nc',
           dirname='.',tlevel=0, too_large_deg=2., opacity=1.0):
    t1 = time.time()

    pathname = find_domain_file(domain_dir,['mask.nc', 'allmeshes.nc'])
    with Dataset(pathname) as f:
        Nd = f.variables['tmask']
        nzm, nym, nxm = Nd.shape[-3:]
        Tsea = Nd[0,:,ys:ye,xs:xe].astype(np.bool)
        if xs is None and xe is None:
            print('correcting sea mask')
            Tsea[:,:,-1] = Tsea[:,:,1]

    pathname = pjoin(dirname, fname)
    if not os.path.exists(pathname):
        sys.exit('cannot find file ',pathname )
    if vble == 'speed':
        velocity = {}
        for component, vname in zip(('U', 'V'),('vozocrtx','vomecrty')):
            pathname = pathname[:-4]+component+'.nc'
            with Dataset(pathname) as f:
                Nd = f.variables[vname]
                try:
                    velocity[component] = Nd[tlevel,:,ys:ye,xs:xe].data
                except:
                    velocity[component] = Nd[tlevel,:,ys:ye,xs:xe]
        nz, ny, nx = Nd.shape[-3:]
        # average onto T points
        T = np.empty([nz, ny, nx ], dtype=Nd.dtype)
        T[:,:,1:] = .5*np.sqrt( (velocity['U'][:,1:,:-1] + velocity['U'][:,1:,1:])**2 +
                          (velocity['V'][:,:-1,1:] + velocity['V'][:,1:,1:])**2 )
        T[:,:,0] = T[:,:,-1]
        if nx == nxm and xs is None and xe is None:
            xs = 1
            xe = nx - 1
        if ny == nym and ys is None and ye is None:
            ys = 1
            ye = ny - 1
        T = T[ys:ye,xs:xe]
    else:
        with Dataset(pathname) as f:
            Nd = f.variables[vble]
            nz, ny, nx = Nd.shape[-3:]
            if nx == nxm and xs is None and xe is None:
                xs = 1
                xe = nx #- 1
            if ny == nym and ys is None and ye is None:
                ys = 1
                ye = ny #- 1
            try:
                T = Nd[tlevel,:,ys:ye,xs:xe].data
            except:
                T = Nd[tlevel,:,ys:ye,xs:xe]

    pathname = find_domain_file(domain_dir,['mesh_hgr.nc', 'allmeshes.nc', coordinate_file])
    with Dataset(pathname) as f:
        fv = f.variables
        Surface.lat = fv['gphit'][0,ys:ye,xs:xe].astype(np.float64)
        Surface.lon = fv['glamt'][0,ys:ye,xs:xe].astype(np.float64)

    pathname = find_domain_file(domain_dir,['mesh_zgr.nc', 'allmeshes.nc', coordinate_file])
    with Dataset(pathname) as f:
        fv = f.variables
        Surface.height = -fv['gdept'][0,:,ys:ye,xs:xe].astype(np.float64)


    Tsea = Tsea[:, ys:ye,xs:xe]
    Surface.kmt = Tsea.astype(int).sum(0)
    T[~Tsea] = np.NaN
    t1, t0 = time.time(), t1

    print('time taken to get data is', t1 - t0, ' s','\n')

    Surface.T = T
    Surface.proj = proj
    Surface.di = 0. #di
    Surface.dj = 0. #dj
    Surface.opacity = opacity
    Surface.too_large = 6e6*(np.pi/180.)*too_large_deg

    processes = max(cpu_count() - 2,1)
    sys.stdout.flush()

    colors = [(0.3, 0.3, 0.8), (0.5, 0.8, 0.3), (0.8, 0.8, 0.3), (0.8, 0.3, 0.3)]
    nvalues = len(values)

    if processes>1 and nvalues>1:
        # create processes workers
        pool = Pool(processes=processes)
        # append output from dopict to xyzs_list for l=0, 1, ...ntraj-1
        xyzfaces_list = pool.map(do_surface,values)
        # wait for all workers to finish & exit parallellized stuff
        pool.close()
    else:
        xyzfaces_list = []
        for value in values:
            xyzfaces_list.append(do_surface(value))
    del Surface.T, Surface.kmt, Tsea, T, Surface.height

    NaN_color = (0.,0.,0.,.0)
    for xyz_faces, color in zip(xyzfaces_list, colors):
        x, y, z, faces = xyz_faces
        tsurf = mlab.triangular_mesh(x, y, z, faces, color = color, opacity=Surface.opacity)
        tsurf.module_manager.scalar_lut_manager.lut.nan_color = NaN_color

if __name__=="__main__":
    from mpl_toolkits.basemap import Basemap
    import lego5
    try:
        os.chdir('/Volumes/Canopus/Programming/NEMO/ARIANE')
    except:
        os.chdir('/Users/agn/Programming/NEMO/ARIANE')
    domain_dir = '../1' #'../025'
    globe = True # False
    if globe:
        map = None
    else:
        map = Basemap(projection='npstere',boundinglat=10,lon_0=270,resolution='l')

    topo = lego5.Topography(globe=globe,domain_dir=domain_dir, map2d=map)
    do_vol('votemper','newP1y1948to1951_TS.nc',[0.],topo.proj, dirname=domain_dir)
    mlab.show()
