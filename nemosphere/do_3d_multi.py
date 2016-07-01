#!/usr/bin/env python
from __future__ import print_function, division
import os
from os.path import join as pjoin
from argparse import ArgumentParser
#
##BSUB -W 00:05
##BSUB -q lotus
##BSUB -n 1
##BSUB -R "rusage[mem=2000]"
##BSUB -o 3d.%I
##BSUB -e 3d.%I
#
class SplitRendering(object):
    def __init__(self, i00, i11, njobs):
        self.i00, self.i11 = i00, i11
        nimages = i11 - i00 + 1
        self.di = nimages // njobs
        self.nfull = nimages - self.di*njobs

    def frame(self,njob):
        if njob <= self.nfull:
            i0 = (njob - 1)*(self.di + 1) + self.i00
            i1 = i0 + self.di
        else:
            i0 = self.nfull*(self.di + 1) + self.i00 + \
            (njob - self.nfull - 1)*self.di
            i1 = i0 + self.di - 1
        return i0, i1


if __name__ == '__main__':
    parser = ArgumentParser(description='break up large number of image creations')
    parser.add_argument('--njobs', type=int, help='no of jobs',
                         dest='njobs', default=1)
    parser.add_argument('--i0', type=int, help='min link #',
                        dest='i0', default=0)
    parser.add_argument('--i1', type=int, help='max link #',
                        dest='i1', default=None)
    parser.add_argument(dest='name', help='name of run script')
    parser.add_argument('--mem', type=int, help='max memory reqd in GB',
                        dest='memory', default=4)
    args = parser.parse_args()

    njob = int(os.environ['LSB_JOBINDEX'])

    split = SplitRendering(args.i0, args.i1, args.njobs)
    # split = SplitRendering(1001, 1027, 3)
    # name = 'run_3d_series_agulhas_salt.bash'
    i0, i1 = split.frame(njob)
    print('doing job # %i with i0 = %i i1 = %i' % (njob, i0, i1))
    os.system('run_3d_series.bash %i %i %s' % (i0, i1, args.name))
