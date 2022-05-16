# Eureka2016SnowAnalysis
Analysing data from the Operation IceBridge 2016 Campaign in coincidence with the ECCC 2016 Snow on Sea Ice Campaign.
Claude de Rijke-Thomas
16th May 2022

Uses a python 3.8.2 environment

To create the environment and install the packages used:
```
conda create -n icyenv python=3.8.2
conda activate icyenv
conda uninstall -c conda-forge cartopy proj geos numpy cython shapely proj pyshp six jupyterlab
conda install -c conda-forge cartopy==0.18
conda uninstall numpy
pip install pyproj==3.3.0
pip install h5py==3.6.0
pip install netCDF4==1.5.8
pip install numba==0.55.1
pip uninstall numpy
pip install numpy==1.20.0
pip install geopandas==0.10.2
pip install git+"https://bitbucket.org/william_rusnack/to-precision/src/master/"
pip install --upgrade jupyterlab
```
