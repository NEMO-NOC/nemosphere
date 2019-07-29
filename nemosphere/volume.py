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

import nemo_rho
from .lego5 import find_domain_file


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


def get_wrap(nx=None, ny=None):
    if nx is not None:
        n = nx
        var = 'i'
    elif ny is not None:
        n = ny
        var = 'j'
    else:
        sys.exit('neither nx nor ny is specified')

    fullwrap = {}
    fullwrap['i'] = (362, 1444, 4322)
    fullwrap['j'] = (292, 1021, 3059)

    core = {}
    core['i'] = (360, 1442, 4320)
    core['j'] = (291, 1020, 3058)

    if n in fullwrap[var]:
        return 'fullwrap'
    elif n in core[var]:
        return 'fullcore'
    else:
        return 'part'
def get_varNd(variable,f):
    vardict = {'theta':['potemp', 'votemper'],
               'S':['vosaline', 'salin'],
               'U':['vozocrtx', 'uo'],
               'V':['vomerty', 'vo']}
    # if variable not a key of vardict, just return as is
    vnames = vardict.get(variable, [variable])
    for vname in vnames:
        if vname in list(f.variables.keys()):
            return f.variables[vname]
    else:
        sys.exit('variables %s are not in file' % ' '.join(vnames))

def stripmask(variable):
    try:
        return variable.data
    except:
        return variable

def do_vol(vble, fname, values, proj,
           xs=None, xe=None, ys=None, ye=None, domain_dir='.',
           coordinate_file = 'coordinates.nc',
           dirname='.',tlevel=0, too_large_deg=2., opacity=1.0):
    t1 = time.time()


    pathname = find_domain_file(domain_dir,['mask.nc', 'allmeshes.nc'])
    with Dataset(pathname) as f:
        Nd = f.variables['tmask']
        nzm, nym, nxm = Nd.shape[-3:]

        if xs is None:
            if get_wrap(nx=nxm) == 'fullwrap':
                xs = 1
            else:
                xs = 0
        if xe is None:
            xe = nym
        if ys is None:
            ys = 0
        if ye is None:
            ye = nym

        print(Nd.shape)
        Tsea = Nd[0,:,ys:ye,xs:xe].astype(np.bool)
        nz, ny, nx = Tsea.shape

    if get_wrap(nx=xe) == 'fullwrap':
        print('correcting sea mask')
        Tsea[:,:,-1] = Tsea[:,:,0]

    pathname = pjoin(dirname, fname)
    if not os.path.exists(pathname):
        sys.exit('cannot find file %s' % pathname )
    if vble == 'speed':
        velocity = {}
        for component in ('U', 'V'):
            pathname = pathname[:-8] + pathname[-8:].replace('U', component)
            with Dataset(pathname) as f:
                Nd = get_varNd(component, f)
                nz, nys, nxs = Nd.shape[-3:]
                if ys > 0 and xs > 0:
                    velocity[component][:,:,:] = data(Nd[tlevel,:,ys-1:ye,xs-1:xe])
                else:
                    velocity[component] = np.empty([nz, ny+1, nx+1])
                    velocity[component][:,1:,1:] = data(Nd[tlevel,:,ys:ye,xs:xe])
                    if get_wrap(nx=nxs) == 'fullcore':
                        velocity[component][:,:,0] = data(Nd[tlevel,:,ys:ye,-1])
                    else:
                        velocity[component][:,0,1:] = velocity[component][:,1,1:]

        # average onto T points
        T = .5*np.sqrt( (velocity['U'][:,1:,:-1] + velocity['U'][:,1:,1:])**2 +
                          (velocity['V'][:,:-1,1:] + velocity['V'][:,1:,1:])**2 )
    elif 'sigma' in vble:
        if xs is None:
            xs=1
        if ys is None:
            ys=1

        print('sigma found')
        TS = {}
        with Dataset(pathname) as f:
            for act_vble in ('theta', 'S'):
                TS[act_vble] =  get_varNd(act_vble, f)[tlevel,:,ys:ye,xs:xe]
                TS[act_vble] = stripmask(TS[act_vble])
        #di, dj = 0, 0
    else:
        if xs is None:
            xs=1
        if ys is None:
            ys=1
        with Dataset(pathname) as f:
            Nd = get_varNd(vble, f)
            nz, ny, nx = Nd.shape[-3:]
            if (ny, nx) != (nym, nxm):
                sys.exit('Dataset %s has different shape %5i %5i to mask file %5i %5i' %
                         (vble, ny, nx, nym, nxm))
            T = data(Nd[tlevel,:,ys:ye,xs:xe])

    if 'sigma' in vble:
        print('sigma found again')
        if TS['theta'].dtype == np.float64:
            sigma = nemo_rho.eos.sigma_n8
        else:
            sigma = nemo_rho.eos.sigma_n4
        ref_depth_km = float(vble[-1])
        neos = 0
        T = sigma(1.e20, ~Tsea.ravel(), TS['theta'].ravel(),
                        TS['S'].ravel(), ref_depth_km, neos ).reshape(Tsea.shape)

    pathname = find_domain_file(domain_dir,['mesh_hgr.nc', 'allmeshes.nc', coordinate_file])
    with Dataset(pathname) as f:
        fv = f.variables
        Surface.lat = fv['gphit'][0,ys:ye,xs:xe].astype(np.float64)
        Surface.lon = fv['glamt'][0,ys:ye,xs:xe].astype(np.float64)

    pathname = find_domain_file(domain_dir,['mesh_zgr.nc', 'allmeshes.nc', coordinate_file])
    with Dataset(pathname) as f:
        fv = f.variables
        vbles = list(fv.keys())
    if 'gdept' in vbles:
        dNd = fv['gdept']
    elif 'gdept_0' in vbles:
        dNd = fv['gdept_0']
        Surface.height = -dNd[0,:,ys:ye,xs:xe].astype(np.float64)


    Tsea = Tsea[:, ys:ye,xs:xe]
    Surface.kmt = Tsea.astype(int).sum(0)
    T[~Tsea] = np.NaN
    t1, t0 = time.time(), t1

    print('time taken to get data is', t1 - t0, ' s','\n')

    Surface.T = T
    print ('Surface.T has shape', Surface.T.shape)
    print('ys, ye=',ys, ye,' xs, xe=',xs, xe)
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
