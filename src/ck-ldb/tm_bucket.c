#include <stdio.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include "tm_tree.h"
#include "tm_bucket.h"
#include "tm_timings.h" 
#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#define random() rand()
#define srandom(x)  srand(x)
#endif

#if __CHARMC__
#include "converse.h"
#else
#define CmiLog2  log2
#endif

#undef DEBUG
bucket_list_t global_bl; 

int tab_cmp(const void* x1,const void* x2){ 
  int *e1,*e2,i1,i2,j1,j2;
  double **tab;
  bucket_list_t bl;

  bl=global_bl;

  e1=((int *)x1);
  e2=((int *)x2);

  
  tab=bl->tab;

  i1=e1[0];
  j1=e1[1];
  i2=e2[0];
  j2=e2[1];

  return tab[i1][j1]>tab[i2][j2]?-1:1;
}


int old_bucket_id(int i,int j,bucket_list_t bucket_list){
  double *pivot,val;
  int n,sup,inf,p;
  pivot=bucket_list->pivot;
  n=bucket_list->nb_buckets;  
  val=bucket_list->tab[i][j];
  
  inf=-1;
  sup=n;

  while(sup-inf>1){
    p=(sup+inf)/2;
    //printf("%f [%d,%d,%d]=%f\n",val,inf,p,sup,pivot[p]);
    if(val<pivot[p]){
      inf=p;
      if(inf==sup)
	inf--;
    }else{
      sup=p;
      if(sup==inf)
	sup++;
    }
  }
  //exit(-1);
  return sup;
}


int bucket_id(int i,int j,bucket_list_t bucket_list){
  double *pivot_tree,val;
  int p,k;
  pivot_tree=bucket_list->pivot_tree;
  val=bucket_list->tab[i][j];



  p=1;
  for(k=0;k<bucket_list->max_depth;k++){
    if(val>pivot_tree[p])
      p=p*2;
    else
      p=p*2+1;
  }

  return  (int)pivot_tree[p];
}
  


void  display_bucket(bucket_t *b){
  printf("\tb.bucket=%p\n",b->bucket);
  printf("\tb.bucket_len=%d\n",(int)b->bucket_len);
  printf("\tb.nb_elem=%d\n",(int)b->nb_elem);

}

void check_bucket(bucket_t *b,double **tab,double inf, double sup,int N){
  int k,i,j;
  for(k=0;k<b->nb_elem;k++){
    i=b->bucket[k].i;
    j=b->bucket[k].j;
    if((tab[i][j]<inf)||(tab[i][j]>sup)){
      printf("[%d] (%d,%d):%f not in [%f,%f]\n",k,i,j,tab[i][j],inf,sup);
      exit(-1);
    }
  }
}

void  display_pivots(bucket_list_t bucket_list){
  int i;

  for(i=0;i<bucket_list->nb_buckets-1;i++){
    printf("pivot[%d]=%f\n",i,bucket_list->pivot[i]);
  }
  printf("\n");
}

void  display_bucket_list(bucket_list_t bucket_list){
  int i;
  double inf,sup;

  display_pivots(bucket_list);

  for(i=0;i<bucket_list->nb_buckets;i++){
    inf=bucket_list->pivot[i];
    sup=bucket_list->pivot[i-1];
    if(i==0)
      sup=DBL_MAX;
    if(i==bucket_list->nb_buckets-1)
      inf=0;
    printf("Bucket %d:\n",i);
    display_bucket(bucket_list->bucket_tab[i]);
    printf("\n");
    check_bucket(bucket_list->bucket_tab[i],bucket_list->tab,inf,sup,bucket_list->N);
  }
  
}

void add_to_bucket(int id,int i,int j,bucket_list_t bucket_list){
  bucket_t *bucket;
  int N,n,size;

  bucket=bucket_list->bucket_tab[id];
  //display_bucket(bucket);
  
  if(bucket->bucket_len==bucket->nb_elem){
    N=bucket_list->N;
    n=bucket_list->nb_buckets;  
    size=N*N/n;
    //display_bucket(bucket);
    bucket->bucket=(coord*)realloc(bucket->bucket,sizeof(coord)*(size+bucket->bucket_len));
    bucket->bucket_len+=size;
#ifdef DEBUG
    printf("malloc/realloc: %d\n",id);
    printf("(%d,%d)\n",i,j);
    display_bucket(bucket);
    printf("\n");
#endif
  }
  
 bucket->bucket[bucket->nb_elem].i=i;
 bucket->bucket[bucket->nb_elem].j=j;
 bucket->nb_elem++;

  //printf("\n");
  //exit(-1);
}

void dfs(int i,int inf,int sup,double *pivot,double *pivot_tree,int depth,int max_depth){
  int p;
  if(depth==max_depth)
    return;

  p=(inf+sup)/2;
  pivot_tree[i]=pivot[p-1];

  dfs(2*i,inf,p-1,pivot,pivot_tree,depth+1,max_depth);
  dfs(2*i+1,p+1,sup,pivot,pivot_tree,depth+1,max_depth);
}

void  built_pivot_tree(bucket_list_t bucket_list){
  double *pivot_tree,*pivot;
  int n,i,k;
  pivot=bucket_list->pivot;
  n=bucket_list->nb_buckets;
  pivot_tree=(double*)malloc(sizeof(double)*2*n);
  bucket_list->max_depth=(int)CmiLog2(n);

  dfs(1,1,n-1,pivot,pivot_tree,0,bucket_list->max_depth);

  k=0;
  for(i=n;i<2*n;i++)
    pivot_tree[i]=k++;

  bucket_list->pivot_tree=pivot_tree;  
  /*
  for(i=0;i<2*n;i++)
    printf("%d:%f\t",i,pivot_tree[i]);
  printf("\n");
  */
}

void fill_buckets(bucket_list_t bucket_list){
  int N,i,j,id;

  N=bucket_list->N;

  for(i=0;i<N;i++){
    for(j=i+1;j<N;j++){
      id=bucket_id(i,j,bucket_list);
      add_to_bucket(id,i,j,bucket_list);
    }
  }
  
  
}

int is_power_of_2(int val){
  int n=1;
  do{
    if(n==val)
      return 1;
    n<<=1;
  }while(n>0);
  return 0;
}


void partial_sort(bucket_list_t *bl,double **tab,int N,int nb_buckets){
  int *sample;
  int i,j,k,n;
  int id;
  double *pivot;
  bucket_list_t bucket_list;


  if(!is_power_of_2(nb_buckets)){
    fprintf(stderr,"Error! Paramater nb_buckets is: %d and should be a power of 2\n",nb_buckets);
    exit(-1);
  }


  bucket_list=(bucket_list_t)malloc(sizeof(_bucket_list_t));

  bucket_list->tab=tab;
  bucket_list->N=N;


  n=pow(nb_buckets,2);
  
  assert(n=N);
  printf("N=%d, n=%d\n",N,n);
  sample=(int*)malloc(2*sizeof(int)*n);
  
  for(k=0;k<n;k++){
    i=random()%(N-2)+1;
    if(i==N-2)
      j=N-1;
    else
      j=random()%(N-i-2)+i+1;
    assert(i!=j);
    assert(i<j);
    assert(i<N);
    assert(j<N);
    sample[2*k]=i;
    sample[2*k+1]=j;
  }
  
  global_bl=bucket_list;
  qsort(sample,n,2*sizeof(int),tab_cmp);
  /*
  for(k=0;k<n;k++){
    i=sample[2*k];
    j=sample[2*k+1];
    printf("%f\n",tab[i][j]);
    }*/
  
  pivot=(double*)malloc(sizeof(double)*nb_buckets-1);
  id=1;
  for(k=1;k<nb_buckets;k++){
    i=sample[2*(id-1)];
    j=sample[2*(id-1)+1];
    id*=2;
    

    /*    i=sample[k*N/nb_buckets]/N;
	  j=sample[k*N/nb_buckets]%N;*/
    pivot[k-1]=tab[i][j];
    //printf("pivot[%d]=%f\n",k-1,tab[i][j]);
  }

  bucket_list->pivot=pivot;
  bucket_list->nb_buckets=nb_buckets;
  built_pivot_tree(bucket_list);
  
  bucket_list->bucket_tab=(bucket_t**)malloc(nb_buckets*sizeof(bucket_t*));
  for(i=0;i<nb_buckets;i++){
    bucket_list->bucket_tab[i]=(bucket_t*)calloc(1,sizeof(bucket_t));
  }

  fill_buckets(bucket_list);
  
  //display_bucket_list(bucket_list);

  bucket_list->cur_bucket=0;
  bucket_list->bucket_indice=0;
  
  free(sample);

  *bl=bucket_list;
}

void next_bucket_elem(bucket_list_t bucket_list,int *i,int *j){
  int N;
  bucket_t *bucket=bucket_list->bucket_tab[bucket_list->cur_bucket];

    //display_bucket_list(bucket_list);
  //printf("nb_elem: %d, indice: %d, bucket_id: %d\n",(int)bucket->nb_elem,bucket_list->bucket_indice,bucket_list->cur_bucket);

  while(bucket->nb_elem<=bucket_list->bucket_indice){

    bucket_list->bucket_indice=0;
    bucket_list->cur_bucket++;
    bucket=bucket_list->bucket_tab[bucket_list->cur_bucket];
      
    //printf("### From bucket %d to bucket %d\n",bucket_list->cur_bucket-1,bucket_list->cur_bucket);
    //printf("nb_elem: %d, indice: %d, bucket_id: %d\n",(int)bucket->nb_elem,bucket_list->bucket_indice,bucket_list->cur_bucket);
    //sleep(1);
  }

  if(!bucket->sorted){
    global_bl=bucket_list;
    qsort(bucket->bucket,bucket->nb_elem,2*sizeof(int),tab_cmp);
    bucket->sorted=1;
  }


  
  N=bucket_list->N;

  *i=bucket->bucket[bucket_list->bucket_indice].i;
  *j=bucket->bucket[bucket_list->bucket_indice].j;
  bucket_list->bucket_indice++;
}


int add_edge_3(double **tab,tree_t *tab_node, tree_t *parent,int i,int j,int N,int *nb_groups){
  //printf("%d <-> %d ?\n",tab_node[i].id,tab_node[j].id);

  if((!tab_node[i].parent) && (!tab_node[j].parent)){
    if(parent){
      parent->child[0]=&tab_node[i];
      parent->child[1]=&tab_node[j];
      tab_node[i].parent=parent;
      tab_node[j].parent=parent;
#ifdef DEBUG
      printf("%d: %d-%d\n",*nb_groups,parent->child[0]->id,parent->child[1]->id);
#endif
      return 1;
    } 
    return 0;
  }
  
  if(tab_node[i].parent && (!tab_node[j].parent)){
    parent=tab_node[i].parent;
    if(!parent->child[2]){
      parent->child[2]=&tab_node[j];
      tab_node[j].parent=parent;
#ifdef DEBUG
      printf("%d: %d-%d-%d\n",*nb_groups,parent->child[0]->id,parent->child[1]->id,parent->child[2]->id);
#endif
      (*nb_groups)++;
    }
    return 0;
  }

  if(tab_node[j].parent && (!tab_node[i].parent)){
    parent=tab_node[j].parent;
    if(!parent->child[2]){
      parent->child[2]=&tab_node[i];
      tab_node[i].parent=parent;
#ifdef DEBUG
      printf("%d: %d-%d-%d\n",*nb_groups,parent->child[0]->id,parent->child[1]->id,parent->child[2]->id);
#endif      
      (*nb_groups)++;
    }
    return 0;
  }

  return 0;
}

int try_add_edge(double **tab,tree_t *tab_node, tree_t *parent,int arity,int i,int j,int N,int *nb_groups){

  assert(i!=j);

  
  switch(arity){
  case 2:
    if(tab_node[i].parent)
      return 0;
    if(tab_node[j].parent)
      return 0;

    parent->child[0]=&tab_node[i];
    parent->child[1]=&tab_node[j];
    tab_node[i].parent=parent;
    tab_node[j].parent=parent;
    
    (*nb_groups)++;

    return 1;
  case 3:
    return add_edge_3(tab,tab_node,parent,i,j,N,nb_groups);
  default:
    fprintf(stderr,"Cannot handle arity %d\n",parent->arity);
    exit(-1);
  }
}


void free_bucket(bucket_t *bucket){
  free(bucket->bucket);
  free(bucket);

}

void free_tab_bucket(bucket_t **bucket_tab,int N){
  int i;
  for(i=0;i<N;i++){
    free_bucket(bucket_tab[i]);
  }
  free(bucket_tab);
}


void free_bucket_list(bucket_list_t bucket_list){

  // Do not free the tab field it is used elsewhere

  free_tab_bucket(bucket_list->bucket_tab,bucket_list->nb_buckets);
  free(bucket_list->pivot);
  free(bucket_list->pivot_tree);
  free(bucket_list);
}

void bucket_grouping(double **tab,tree_t *tab_node, tree_t *new_tab_node, int arity,int N, int M,long int k){
  bucket_list_t bucket_list;
  double duration;
  int l,i,j,nb_groups;
  double val=0;
 
  TIC;
  partial_sort(&bucket_list,tab,N,8);
  duration=TOC;
  printf("Partial sorting=%fs\n",duration);  

  display_pivots(bucket_list);

 
  TIC;
  l=0;
  i=0;
  nb_groups=0;
  while(l<M){
    next_bucket_elem(bucket_list,&i,&j);
    if(try_add_edge(tab,tab_node,&new_tab_node[l],arity,i,j,N,&nb_groups)){
      l++;
    }
  }

#ifdef DEBUG
  printf("l=%d,nb_groups=%d\n",l,nb_groups);
#endif

  while(nb_groups<M){
    next_bucket_elem(bucket_list,&i,&j);
    try_add_edge(tab,tab_node,NULL,arity,i,j,N,&nb_groups);
  }

#ifdef DEBUG
  printf("l=%d,nb_groups=%d\n",l,nb_groups);
#endif

  for(l=0;l<M;l++){
    update_val(tab,&new_tab_node[l],N);      
    val+=new_tab_node[l].val;
  }


      


  duration=TOC;
  printf("Grouping =%fs\n",duration);  

  printf("Bucket: %d, indice:%d\n",bucket_list->cur_bucket,bucket_list->bucket_indice);

  printf("val=%f\n",val);
  free_bucket_list(bucket_list);

  //  exit(-1);

  //  display_grouping(new_tab_node,M,arity,val);
  

}

