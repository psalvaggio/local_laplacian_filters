# README #

Implementation of the Local Laplacian Filters image processing algorithm in C++ using OpenCV. The algorithm is described here:

Paris, Sylvain, Samuel W. Hasinoff, and Jan Kautz. "Local Laplacian filters: edge-aware image processing with a Laplacian pyramid." ACM Trans. Graph. 30.4 (2011): 68.

The project is built using CMake


```
#!bash
mkdir build && cd build
cmake ..
make
```

The code has currently been tested for detail enhancement and reduction. Tone mapping is untested, but will be soon.