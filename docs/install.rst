Installation
------------

Prerequisites
=============
Currently requires python 2.7 because of the dependency on mayavi. 
Also requires numpy, scipy, scikits-image, basemap and numba.

Because of the heavy dependencies (mayavi and numba) it may be wisest
to first install numpy, scipy, scikits-image, basemap, numba and mayavi
using a package manager like macports or conda. So with macports after
installing macports itself according to the instructions at
https://www.macports.org/install.php do:

.. highlight:: bash

::

 sudo port install py27-numpy py27-scipy py27-scikit-image \
 py27-basemap py27-numba py27-mayavi py27-pip

Or with conda first install either the anaconda or miniconda python-2
distributions (see
e.g. http://conda.pydata.org/docs/install/quick.html).
Then

.. code-block:: bash

 conda install numpy scipy scikit-image basemap numba mayavi pip


nemosphere installation
========================


User install
++++++++++++++++

If either (i) the
python is a system python to whose site-packages directory you do not
have write access or (ii) you simply wish to keep the packages you install
manually separate from the macports/conda packages, you will need to
do a user install

.. code-block:: bash

 pip install --user nemosphere

This installs the nemosphere package into your personal python
``site-packages`` directory (where it will be on the python ``sys.path`` and
so will be importable) and the key command-line script ``3ddriver``
into your personal python ``bin`` directory. You can find out where
these directories are on your system by doing

.. code-block:: bash

 python -m site

The resulting output will give (e.g. for a
macports python installation):

.. code-block:: bash

 ~ $ python -m site
 sys.path = [
    ...,
 ]
 USER_BASE: '/Users/your_id/Library/Python/2.7' (exists)
 USER_SITE: '/Users/your_id/Library/Python/2.7/lib/\
                                   python/site-packages' (exists)
 ENABLE_USER_SITE: True

where ``USER_SITE`` is your personal ``site-packages`` directory, and
``USER_BASE/bin`` the personal python ``bin`` directory, into which
``3ddriver`` is installed. For a {mini,ana}conda distribution in
linux ``USER_BASE`` will instead be ``~/.local``.

You need to make sure that ``3ddriver`` is on your ``PATH``. Try ``which 3ddriver``; if this returns a path ``$USER_BASE/bin/3ddriver`` then you are OK.
However, if it returns a blank line, you need to put ``3ddriver`` onto your ``PATH``. If you only have python 2, you can simply add ``USER_BASE/bin`` to
your ``PATH``

.. code-block:: bash

 export PATH=~/Library/Python/2.7/bin:$PATH
 
otherwise you can just explicitly link ``3ddriver`` to a directory which is on your ``PATH``:

.. code-block:: bash

 export PATH=~/bin:$PATH
 ln -s /Library/Python/2.7/bin/3ddriver ~/bin


Development user install
+++++++++++++++++++++++++++

Or if you plan to modify the code, download it directly from github
and do a developer install:

.. code-block:: bash

 git clone https://github.com/NEMO-NOC/nemosphere.git
 pip install --user -e nemosphere

This way creates links from your git nemosphere directory into your (personal)
site-packages directory, so that changes you make to code in the git
directory are immediately reflected in the modules imported by python
and the 3ddriver command-line script. You need to make sure
``3ddriver`` is on your ``PATH`` in the same way as for the standard
user install.

System install
++++++++++++++++

If you want to make the package available to more than one user on your system, you may wish to do a system install.

.. code-block:: bash

 pip install nemosphere


This then installs the nemosphere package into the standard python
``site-packages`` directory where it will on the python ``sys.path`` and
so will be importable and the key command-line script ``3ddriver``
into the standard python ``bin`` directory.

.. code-block:: bash

 # For ana/miniconda these are
 ~/miniconda/lib/python2.7/site-packages
 ~/miniconda/bin
 # or for macports python
 /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib\
                               /python2.7/site-packages
 /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin

{Ana,Mini}conda by default puts the  ``~/{ana,mini}conda/bin`` directory
onto your path, so that ``3ddriver`` will be available. Macports may
not automatically do this, so you may wish
to symbolically link it to a directory which is already on your
``PATH``; e.g.

.. code-block:: bash

 export PATH=~/bin:$PATH
 # when you run this remove blank spaces from the second line!
 ln -s /opt/local/Library/Frameworks/Python.framework/Versions\
                                  /2.7/bin/3ddriver  ~/bin
