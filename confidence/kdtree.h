#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct KDTree;

KDTree* kdtree_build(const float* data, int n, int dim);
float   kdtree_query_nn(const KDTree* tree, const float* query, int* out_idx);
void    kdtree_free(KDTree* tree);

#ifdef __cplusplus
}
#endif
