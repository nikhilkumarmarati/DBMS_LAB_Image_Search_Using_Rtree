/*
c++ -O3 -Wall -shared -std=c++17 -fPIC     $(python3 -m pybind11 --includes)     rtree.cpp -o rtree.so     $(python3-config --ldflags)
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rtree.h"  // Include your existing header

namespace py = pybind11;

// Expose Rtree class to Python
PYBIND11_MODULE(rtree, m) {
    py::class_<RTree>(m, "Rtree")
        .def(py::init<>())  // Bind constructor
        .def("load", &RTree::load, "Load R-tree from file")  // Expose the load method
        .def("kNearestNeighbors", &RTree::kNearestNeighbors, "Find k nearest neighbors");
}
