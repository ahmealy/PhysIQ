/**
 * kdtree_bind.cpp — pybind11 Python bindings for KDTree.
 *
 * Exposed as: from _kdtree import KDTree
 *
 * Interface:
 *   KDTree(data: np.ndarray[float32, shape=(N, dim)])
 *   .query(point: np.ndarray[float32, shape=(1, dim)], k: int = 1)
 *       -> (distance: float, index: int)
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kdtree.h"

namespace py = pybind11;

class PyKDTree {
public:
    explicit PyKDTree(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto buf = data.request();
        if (buf.ndim != 2)
            throw std::runtime_error("data must be 2-D array [N, dim]");
        n_   = static_cast<int>(buf.shape[0]);
        dim_ = static_cast<int>(buf.shape[1]);
        tree_ = kdtree_build(static_cast<const float*>(buf.ptr), n_, dim_);
    }

    ~PyKDTree() {
        if (tree_) kdtree_free(tree_);
    }

    // query(point, k=1) -> (distance, index)
    // point shape: [1, dim]  (same as scipy interface: query(point.reshape(1, -1), k=1))
    std::pair<float, int> query(
        py::array_t<float, py::array::c_style | py::array::forcecast> point,
        int /*k*/ = 1)
    {
        auto buf = point.request();
        // Validate total element count matches dim_
        int total = 1;
        for (auto s : buf.shape) total *= static_cast<int>(s);
        if (total != dim_)
            throw std::runtime_error(
                "query point has wrong dimension: expected " + std::to_string(dim_) +
                ", got " + std::to_string(total));
        const float* ptr = static_cast<const float*>(buf.ptr);
        int best_idx = -1;
        float dist = kdtree_query_nn(tree_, ptr, &best_idx);
        return {dist, best_idx};
    }

private:
    KDTree* tree_ = nullptr;
    int n_ = 0, dim_ = 0;
};

PYBIND11_MODULE(_kdtree, m) {
    m.doc() = "C++ k-d tree for nearest-neighbor confidence scoring";
    py::class_<PyKDTree>(m, "KDTree")
        .def(py::init<py::array_t<float>>(),
             py::arg("data"),
             "Build k-d tree from [N, dim] float32 array")
        .def("query",
             &PyKDTree::query,
             py::arg("point"),
             py::arg("k") = 1,
             "Query nearest neighbor. Returns (distance, index).");
}
