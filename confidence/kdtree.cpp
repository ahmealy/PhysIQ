/**
 * kdtree.cpp — Standard k-d tree on Euclidean distance in R^d.
 *
 * Build: median split on alternating dimensions (standard k-d tree).
 * Search: branch-and-bound with pruning — only recurse into a subtree if it
 *         could contain a closer point than the current best.
 */
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "kdtree.h"

// ── Node ─────────────────────────────────────────────────────────────────────

struct KDNode {
    int   idx;        // index into data array (-1 for internal nodes)
    int   split_dim;
    float split_val;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
};

// ── KDTree implementation ─────────────────────────────────────────────────────

struct KDTreeImpl {
    std::vector<float> data;   // flat [n * dim]
    int n, dim;
    KDNode* root = nullptr;

    // Pool allocator to avoid per-node heap fragmentation
    std::vector<KDNode> pool;
    int pool_pos = 0;

    KDNode* alloc_node() {
        if (pool_pos >= (int)pool.size()) {
            pool.resize(pool.size() == 0 ? 1 : pool.size() * 2);
        }
        KDNode* node = &pool[pool_pos++];
        node->left = node->right = nullptr;
        node->idx = -1;
        return node;
    }

    const float* point(int i) const { return data.data() + i * dim; }

    float sq_dist(const float* a, const float* b) const {
        float d = 0.0f;
        for (int k = 0; k < dim; ++k) {
            float diff = a[k] - b[k];
            d += diff * diff;
        }
        return d;
    }

    KDNode* build(std::vector<int>& indices, int depth) {
        if (indices.empty()) return nullptr;
        KDNode* node = alloc_node();
        if (indices.size() == 1) {
            node->idx = indices[0];
            return node;
        }
        int axis = depth % dim;
        // Partial sort to find median
        int mid = indices.size() / 2;
        std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
            [&](int a, int b) { return point(a)[axis] < point(b)[axis]; });

        node->split_dim = axis;
        node->split_val = point(indices[mid])[axis];
        node->idx       = indices[mid];

        std::vector<int> left_idx(indices.begin(), indices.begin() + mid);
        std::vector<int> right_idx(indices.begin() + mid + 1, indices.end());
        node->left  = build(left_idx,  depth + 1);
        node->right = build(right_idx, depth + 1);
        return node;
    }

    void search(KDNode* node, const float* query,
                float& best_sq, int& best_idx) const {
        if (node == nullptr) return;

        if (node->idx >= 0) {
            float d = sq_dist(query, point(node->idx));
            if (d < best_sq) { best_sq = d; best_idx = node->idx; }
        }

        if (node->left == nullptr && node->right == nullptr) return;

        int axis = node->split_dim;
        float diff = query[axis] - node->split_val;
        KDNode* near = (diff <= 0) ? node->left : node->right;
        KDNode* far  = (diff <= 0) ? node->right : node->left;

        search(near, query, best_sq, best_idx);

        // Prune: only search far side if the splitting hyperplane is within best_sq
        if (diff * diff < best_sq) {
            search(far, query, best_sq, best_idx);
        }
    }
};

// ── Public C API (called from pybind11 binding) ───────────────────────────────

KDTree* kdtree_build(const float* data, int n, int dim) {
    auto* impl = new KDTreeImpl();
    impl->n    = n;
    impl->dim  = dim;
    impl->data.assign(data, data + n * dim);
    impl->pool.reserve(2 * n);   // pre-allocate node pool

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    impl->root = impl->build(indices, 0);
    return reinterpret_cast<KDTree*>(impl);
}

float kdtree_query_nn(const KDTree* tree, const float* query) {
    auto* impl = reinterpret_cast<const KDTreeImpl*>(tree);
    float best_sq = std::numeric_limits<float>::infinity();
    int   best_idx = -1;
    impl->search(impl->root, query, best_sq, best_idx);
    return std::sqrt(best_sq);
}

void kdtree_free(KDTree* tree) {
    delete reinterpret_cast<KDTreeImpl*>(tree);
}
