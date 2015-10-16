#!/usr/bin/env python
from __future__ import print_function
import os, sys, platform
from argparse import ArgumentParser
import numpy as np
import numpy.ma as ma
import time
import resource

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from mayavi.mlab import show, view, gcf, savefig, options

from tvtk.api import tvtk
from tvtk.common import configure_input_data

import lego5
import traj
import volume
import dots

if __name__ == '__main__':
    parser = ArgumentParser(description=
    """
    produce lego-block topography & optionally trajectories and/or volumes
     e.g. 3ddriver.py --jlo <lowest j> -d  <directory for metric files>
    -t <trajectory_file>
    -s <file_with_field_fieldname> --field <fieldname> --levels 35. 35.05
    """)
    parser.add_argument('-b',dest='bounds',help='ilo ihi jlo jhi', type=int, nargs= '*',default=None)
    parser.add_argument('--ilo',dest='ilo',help='ilo; overrrides value in bounds', type=int, default=None)
    parser.add_argument('--jlo',dest='jlo',help='jlo; overrrides value in bounds', type=int, default=None)
    parser.add_argument('--ihi',dest='ihi',help='ihi; overrrides value in bounds', type=int, default=None)
    parser.add_argument('--jhi',dest='jhi',help='jhi; overrrides value in bounds', type=int, default=None)

    parser.add_argument('--bathy',dest='bathymetry_file',help='bathymetry file if not bathy_meter.nc',
                        default='bathy_meter.nc')
    parser.add_argument('--outfile','-o', dest='outfile',help='output file',
                        default=None)
    parser.add_argument('--no_display', dest='no_display',help='if set do not display window',
                        action='store_true', default=False)
    parser.add_argument('--coords',dest='coordinate_file',help='coordinate file if not coordinates.nc or mesh_hgr.nc',
                        default='coordinates.nc')
    parser.add_argument('--domain','-d',dest='domain_dir',
                        help='directory of bathymetry & coordinates',
                        default='./')
    parser.add_argument('--bottom',dest='bottom',type=float,
                         help='(positive) depth where colorscale saturates to deepest value',
                        default=6000.)
    parser.add_argument('--globe','-g', dest='globe',action='store_true',
                         help='do globe', default=False)
    parser.add_argument('--cmap', dest='cmap', help='colormap for topography',
                         default='gist_earth', choices=['gist_earth', 'gist_gray'])
    parser.add_argument('--traj','-t', dest='traj', help='file with trajectories', default=None)
    parser.add_argument('--size', dest='size_in_pixels', help='xsize & ysize of image (in pixels)', type=int, nargs=2,
                        default=(1024,768))
    parser.add_argument('--dots', dest='dots', help='file with trajectories to so dotplots', default=None)
    parser.add_argument('--icb', dest='icb',action='store_true',
                         help='treat as iceberg trajectories', default=False)
    parser.add_argument('--traj_numbers',dest='traj_numbers', type=int, nargs= 3, help='trajectory_numbers',
                         default=[None, None, None])
    parser.add_argument('--times',dest='times', type=int, nargs='*', help='output times',
                         default=[-1])
    parser.add_argument('--traj_cut',dest='traj_cut', type=int, help='last trajectory time level to read',
                         default=None)
    parser.add_argument('--traj_threshold',dest='threshold', type=float,
                         help='threshold in deg for shortening trajectories', default=0.01)
    parser.add_argument('--passes',dest='passes', help='# of passes for path simplification using quick_simplify',
                        type=int, default=50)
    parser.add_argument('--surf','-s', dest='surf_file', help='file with data for 3d surface plot', default=None)
    parser.add_argument('--surf_dir', dest='surf_dir', help='directory with surf_file', default='.')
    parser.add_argument('--field','-f', dest='field', help='field for 3d surface plot', default='votemper')
    parser.add_argument('--opacity', dest='opacity', help='opacity for 3d surface plot', default=0.7)
    parser.add_argument('--camera', dest='camera', help='specify camera angles', default=None)
    parser.add_argument('--levels', dest='levels', type=float, nargs= '*',
                         help='field levels for 3d surface plot', default=[0.])
    parser.add_argument('--stride', dest='stride', type=int,
                         help='every stride-th trajectory plotted', default=1)
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
        map2d = None
    else:
        # Use a basemap projection; see http://matplotlib.org/basemap/users/mapsetup.html
        # Lambert conformal
        # map2d = Basemap(llcrnrlon=-95.,llcrnrlat=1.,urcrnrlon=80.,urcrnrlat=80.,\
        #         rsphere=(6378137.00,6356752.3142),\
        #         resolution='l',area_thresh=1000.,projection='lcc',\
        #         lat_1=50.,lon_0=-35.)
        # Orthographic (still won't work)
        # map2d = Basemap(projection='ortho',lat_0=50.,lon_0=-35.)
        # Mollweide
        # map2d = Basemap(projection='moll',lon_0=0,resolution='c')
        # N Polar stereographic
        map2d = Basemap(projection='npstere',boundinglat=10,lon_0=270,resolution='l')

    if args.no_display:
        options.offscreen = True

    topo = lego5.Topography(xs=xs, xe=xe, ys=ys, ye=ye,
                     domain_dir=args.domain_dir, bathymetry_file=args.bathymetry_file,
                     coordinate_file= args.coordinate_file,
                     bottom = args.bottom, cmap = args.cmap, map2d = map2d, globe = args.globe,
                     size_in_pixels = args.size_in_pixels)

    if args.traj is not None:
        traj.do_trajectories(args.traj, args.traj_numbers, topo, icb=args.icb,
                        xs=xs, xe=xe, ys=ys, ye=ye,
                        traj_cut=args.traj_cut,
                        threshold_deg=args.threshold, passes=args.passes)

    if args.dots is not None:
        dots.do_dots(args.dots, args.times, topo, stride=args.stride)

    if args.surf_file is not None:
        volume.do_vol(args.field, args.surf_file,args.levels,topo.proj,
                        xs=xs, xe=xe, ys=ys, ye=ye, domain_dir=args.domain_dir,
                        dirname=args.surf_dir, opacity=args.opacity)

    scene = gcf()

    if args.camera is not None:
        scene.scene.camera.position = [26061765.205169372, -4221645.3109468669, 11513670.945417598]
        scene.scene.camera.focal_point = [16434449.20815997, 6627234.1068188809, -1408413.5745141534]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.5263926738428919, 0.4160622373626518, 0.74148699756996095]
        scene.scene.camera.clipping_range = [6874912.2407720806, 34807141.917231418]
        scene.scene.camera.compute_view_plane_normal()

    if args.outfile is not None:
        if args.no_display:

            ren = scene.scene._renderer

            #create new off screen renderwindow
            rw = tvtk.RenderWindow(size=args.size_in_pixels, off_screen_rendering=1)
            rw.add_renderer(ren)

            #antialiasing
            rw.aa_frames = 8

            #and save
            w2if = tvtk.WindowToImageFilter()
            w2if.magnification = scene.scene.magnification
            w2if.input = rw
            ex = tvtk.PNGWriter()
            ex.file_name = args.outfile
            configure_input_data(ex, w2if.output)
            w2if.update()
            ex.write()
        else:
            savefig(figure=scene, filename=args.outfile, size=args.size_in_pixels)
    if not args.no_display:
        show()

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
