#ifndef __BUCKET_H__
#define __BUCKET_H__

typedef struct{
  int i;
  int j;
}coord;

typedef struct{
  coord * bucket; /*store i,j*/
  size_t bucket_len; //allocated size in the heap
  size_t nb_elem; // number of usefull elements (nb_elem should be lower than bucket_len)
  int sorted;
}bucket_t;

typedef struct{
  bucket_t **bucket_tab;
  size_t nb_buckets;
  double **tab;
  int N;//length of tab
  //For iterating over the buckets
  int cur_bucket;
  int bucket_indice;
  double *pivot;
  double *pivot_tree;
  int max_depth;
}_bucket_list_t;

typedef _bucket_list_t *bucket_list_t;

void bucket_grouping(double **tab,tree_t *tab_node, tree_t *new_tab_node, int arity,int N, int M,long int k);
int try_add_edge(double **tab,tree_t *tab_node, tree_t *parent,int arity,int i,int j,int N,int *nb_groups);
#endif
