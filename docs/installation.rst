.. highlight:: shell

============
Installation
============


Stable release
--------------

To install lightcone, run this command in your terminal:

.. code-block:: console

    $ pip install lightcone

This is the preferred method to install lightcone, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for lightcone can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/sibirrer/lightcone

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/sibirrer/lightcone/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/sibirrer/lightcone
.. _tarball: https://github.com/sibirrer/lightcone/tarball/master

FFMPEG
------

FFMPEG installation is required to create MP4 files of the animated lightcurves.

* Free download: https://ffmpeg.org/download.html.

After downloading, make sure you include the FFMPEG binary directory (e.g. C:\\users\\user\\ffmpeg\\bin) to your PATH.

On Windows you can edit your environment variables and edit your Path to add to it. You may want to restart your PC after doing so too.

Paltas
------

To install Paltas, run this command in your terminal:

.. code-block:: console

    $ pip install paltas

Lightcone currently uses Paltas version 0.1.1.

If using Paltas with Lightcone, the following catalog should be downloaded:

* COSMOS 23.5 Magnitude Catalog: https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data

After downloading and unzipping the catalog, make sure you go into the config file (e.g. test_config.py) and assign the cosmos_folder variable to the path name of where you unzipped the catalog.
