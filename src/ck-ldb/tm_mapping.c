#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <ctype.h>
#include <math.h>
#include <assert.h> 
#include "tm_mapping.h"
#include "tm_timings.h"
#include "tm_tree.h"
#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#define random() rand()
#define srandom(x)  srand(x)
#endif
 
#define TEST_ERROR(n) {if(n!=0){fprintf(stderr,"Error %d Line %d\n",n,__LINE__);exit(-1);}}
#undef DEBUG

#define LINE_SIZE 1000000

typedef struct{
  int  val;
  long key;
}hash_t;


typedef struct{
  double val;
  int key1;
  int key2;
}hash2_t;


int nb_nodes(tm_topology_t *topology){
  return topology->nb_nodes[topology->nb_levels-1];
}





void free_topology(tm_topology_t *topology){
   int i;
  for(i=0;i<topology->nb_levels;i++)
    free(topology->node_id[i]);
  free(topology->node_id);
  free(topology->nb_nodes);
  free(topology->arity);
  free(topology);
}

double print_sol(int N,int *Value,double **comm, double **arch){
  double sol;
  int i,j;
  double a,c;

  sol=0;
  for (i=0;i<N;i++){
    for (j=i+1;j<N;j++){
      c=comm[i][j];
      a=arch[Value[i]][Value[j]];
      //printf("T_%d_%d %f/%f=%f\n",i,j,c,a,c/a);
      sol+=c/a;
    }
  }
  for (i = 0; i < N; i++) {
    printf("%d", Value[i]);
    if(i<N-1)
      printf(",");
      
  }
  printf(" : %g\n",sol);

  return sol;
}


void print_1D_tab(int *tab,int N){
  int i;
  for (i = 0; i < N; i++) {
    printf("%d", tab[i]);
    if(i<N-1)
      printf(",");
  }
  printf("\n");

}

int nb_lines(char *filename){
  FILE *pf;
  char  line[LINE_SIZE];
  int N=0;

  if(!(pf=fopen(filename,"r"))){
      fprintf(stderr,"Cannot open %s\n",filename);
      exit(-1);
  }

  while(fgets(line,LINE_SIZE,pf))
    N++;

  printf("N=%d\n",N);

  fclose(pf);
  return N;
}

void init_comm(char *filename,int N,double **comm){
  int i,j;
  FILE *pf;
  char *ptr;
  char  line[LINE_SIZE];

  if(!(pf=fopen(filename,"r"))){
    fprintf(stderr,"Cannot open %s\n",filename);
    exit(-1);
  }


  j=-1;
  i=0;
  while(fgets(line,LINE_SIZE,pf)){

  
    char *l=line;
    j=0;
    //printf("%s|",line);
    while((ptr=strtok(l," \t"))){
      if((ptr[0]!='\n')&&(!isspace(ptr[0]))&&(*ptr)){
	comm[i][j]=atof(ptr);
	//printf ("comm[%d][%d]=%f|%s|\n",i,j,comm[i][j],ptr);
	j++;
      }
    }
    if(j!=N){
      fprintf(stderr,"Error at %d %d (%d!=%d)for %s\n",i,j,j,N,filename);
      exit(-1);
    }
    i++;
  }
  if(i!=N){
    fprintf(stderr,"Error at %d %d for %s\n",i,j,filename);
    exit(-1);
  }

  /*  printf("%s:\n",filename);
  for(i=0;i<N;i++){
    for(j=0;j<N;j++){
      printf("%6.1f ",comm[i][j]);
    }
    printf("\n");
    } */
  fclose(pf);
}    

int  build_comm(char *filename,double ***pcomm){
  int i;
  int N;
  double **comm;
  N=nb_lines(filename);
  comm=(double**)malloc(N*sizeof(double*));
  for(i=0;i<N;i++)
    comm[i]=(double*)malloc(N*sizeof(double));
  init_comm(filename,N,comm);
  *pcomm=comm;
  return N;
}

void map_Packed(tm_topology_t *topology,int N,int *Value){
  int i,depth;

  depth=topology->nb_levels-1;

  for(i=0;i<N;i++){
    //printf ("%d -> %d\n",objs[i]->os_index,i);
    Value[i]=topology->node_id[depth][i];
  }

}

void map_RR(int N,int *Value){
  int i;

  for(i=0;i<N;i++){
    //printf ("%d -> %d\n",i,i);
    Value[i]=i;
  }

}



int hash_asc(const void* x1,const void* x2){ 

  hash_t *e1,*e2;

  e1=((hash_t*)x1);
  e2=((hash_t*)x2);

  
  return e1->key<e2->key?-1:1;
} 


int *generate_random_sol(tm_topology_t *topology,int N,int level,int seed){
  hash_t *hash_tab;
  int *sol,i;
  int *nodes_id;

  nodes_id=topology->node_id[level];


  hash_tab=(hash_t*)malloc(sizeof(hash_t)*N);
  sol=(int*)malloc(sizeof(int)*N);
  
  srandom(seed);
  
  for(i=0;i<N;i++){
    hash_tab[i].val=nodes_id[i];
    hash_tab[i].key=random();
  }
  
  qsort(hash_tab,N,sizeof(hash_t),hash_asc);
  for(i=0;i<N;i++){
    sol[i]=hash_tab[i].val;
  }
  free(hash_tab);
  return sol;
}


double eval_sol(int *sol,int N,double **comm, double **arch){
  double res;
  int i,j;
  double a,c;

  res=0;
  for (i=0;i<N;i++){
    for (j=i+1;j<N;j++){
      c=comm[i][j];
      a=arch[sol[i]][sol[j]];
      res+=c/a;
    }
  }
  return res;
}

void exchange(int *sol,int i,int j){
  int tmp;
  tmp=sol[i];
  sol[i]=sol[j];
  sol[j]=tmp;
}

double gain_exchange(int *sol,int l,int m,double eval1,int N,double **comm, double **arch){
  double eval2;
  if(l==m)
    return 0;
  exchange(sol,l,m);
  eval2=eval_sol(sol,N,comm,arch);
  exchange(sol,l,m);
  return eval1-eval2;
}

void select_max(int *l,int *m,double **gain,int N,int *state){
  int i,j;
  double max;
  max=-DBL_MAX;
  
  for(i=0;i<N;i++){
    if(!state[i]){
      for(j=0;j<N;j++){
	  if((i!=j)&&(!state[j])){
	    if(gain[i][j]>max){
	      *l=i;*m=j;
	      max=gain[i][j];
	    }
	  }
      }
    }
  }
  
}

void compute_gain(int *sol,int N,double **gain,double **comm, double **arch){
  int i,j;
  double eval1;
  eval1=eval_sol(sol,N,comm,arch);
  for(i=0;i<N;i++){
    for(j=0;j<=i;j++){
      gain[i][j]=gain[j][i]=gain_exchange(sol,i,j,eval1,N,comm,arch);
    }
  } 
}



/* Randomized Algorithm of
Hu Chen, Wenguang Chen, Jian Huang ,Bob Robert,and H.Kuhn. Mpipp: an automatic profile-guided
parallel process placement toolset for smp clusters and multiclusters. In
Gregory K. Egan and Yoichi Muraoka, editors, ICS, pages 353-360. ACM, 2006.
 */

void map_MPIPP(tm_topology_t *topology,int nb_seed,int N,int *Value,double **comm, double **arch){
  int *sol;
  int *state;
  double **gain;
  int **history;
  double *temp;
  int i,j,t,l=0,m=0,loop=0,seed=0;
  double max,sum,best_eval,eval;

  
  gain=(double**)malloc(sizeof(double*)*N);
  for(i=0;i<N;i++){
    gain[i]=(double*)malloc(sizeof(double)*N);  
    if(!gain[i]){
    }
  }
  history=(int**)malloc(sizeof(int*)*N);
  for(i=0;i<N;i++)
    history[i]=(int*)malloc(sizeof(int)*3);

  state=(int*)malloc(sizeof(int)*N);
  temp=(double*)malloc(sizeof(double)*N);

  sol=generate_random_sol(topology,N,topology->nb_levels-1,seed++);
  for(i=0;i<N;i++)
    Value[i]=sol[i];
  
  best_eval=DBL_MAX;
  while(seed<=nb_seed){
    loop=0;
    do{

      for(i=0;i<N;i++){
	state[i]=0;
	//printf("%d ",sol[i]);
      }
      //printf("\n");
      compute_gain(sol,N,gain,comm,arch);
  
      //display_tab(gain,N);
      //exit(-1);
      for(i=0;i<N/2;i++){
	select_max(&l,&m,gain,N,state);
	//printf("%d: %d <=> %d : %f\n",i,l,m,gain[l][m]);
	state[l]=1;state[m]=1;
	exchange(sol,l,m);
	history[i][1]=l;history[i][2]=m;
	temp[i]=gain[l][m];
	compute_gain(sol,N,gain,comm,arch);
      }

      t=-1;
      max=0;
      sum=0;
      for(i=0;i<N/2;i++){
	sum+=temp[i];
	if(sum>max){
	  max=sum;
	  t=i;
	}
      }
      /*for(j=0;j<=t;j++)
	printf("exchanging: %d with %d for gain: %f\n",history[j][1],history[j][2],temp[j]); */
      for(j=t+1;j<N/2;j++){
	exchange(sol,history[j][1],history[j][2]);
	//printf("Undoing: %d with %d for gain: %f\n",history[j][1],history[j][2],temp[j]); 
      }
      //printf("max=%f\n",max);

      /*for(i=0;i<N;i++){
	printf("%d ",sol[i]);
	}
	printf("\n");*/
      eval=eval_sol(sol,N,comm,arch);
      if(eval<best_eval){
	best_eval=eval;
	for(i=0;i<N;i++)
	  Value[i]=sol[i];
	//print_sol(N);
      }
    

    }while(max>0);
    
    if (sol != NULL) {
      free(sol);
      sol = NULL;
    }
    sol=generate_random_sol(topology,N,topology->nb_levels-1,seed++);

  }
  if (sol != NULL) {
    free(sol);
    sol = NULL;
  }
  free(state);
  free(temp);

  for(i=0;i<N;i++){
    free(history[i]);
    free(gain[i]);
  }
  free(history);
  free(gain);
}
  



void map_tree(tree_t* t1,tree_t *t2){
  /*  double x1,x2;
  if((!t1->left)&&(!t1->right)){
    printf ("%d -> %d\n",t1->id,t2->id);
    Value[t2->id]=t1->id;
   return;
  }
  x1=t2->right->val/t1->right->val+t2->left->val/t1->left->val;
  x2=t2->left->val/t1->right->val+t2->right->val/t1->left->val;
  if(x1<x2){
    map_tree(t1->left,t2->left);
    map_tree(t1->right,t2->right);
  }else{
    map_tree(t1->right,t2->left);
    map_tree(t1->left,t2->right);
    }*/
}

void depth_first(tree_t *comm_tree, int *proc_list,int *i){
  int j;
  if(!comm_tree->child){
    proc_list[(*i)++]=comm_tree->id;
    return;
  }

  for(j=0;j<comm_tree->arity;j++){
    depth_first(comm_tree->child[j],proc_list,i);
  }
}

int nb_leaves(tree_t *comm_tree){
  int n=0,j;

  if(!comm_tree->child){
    return 1;
  }

  for(j=0;j<comm_tree->arity;j++){
    n+=nb_leaves(comm_tree->child[j]);
  }
  return n;
}




/*Map topology to cores: 
 sigma_i is such that  process i is mapped on core sigma_i
 k_i is such that core i exectutes process k_i

 size of sigma is the number of process
 size of k is the number of cores/nodes

 We must have numbe of process<=number of cores

 k_i =-1 if no process is mapped on core i
*/
void map_topology(tm_topology_t *topology,tree_t *comm_tree,int nb_proc,int level,
		  int *sigma, int *k){
  int *nodes_id;
  int N;
  int *proc_list,i,l;
  int M;
  int block_size;

 
  M=nb_leaves(comm_tree);
  printf("nb_leaves=%d\n",M);



  nodes_id=topology->node_id[level];
  N=topology->nb_nodes[level];
  //printf("level=%d, nodes_id=%p, N=%d\n",level,nodes_id,N);


  //printf("N=%d,nb_proc=%d\n",N,nb_proc);
  /* The number of node at level "level" in the tree should be equal to the number of processors*/
  assert(N==nb_proc);


  proc_list=(int*)malloc(sizeof(int)*M);
  i=0;
  depth_first(comm_tree,proc_list,&i);

  l=0;
  for(i=0;i<M;i++){
    //printf ("%d\n",proc_list[i]);
  }


  block_size=M/N;


  if(k){/*if we need the k vector*/
    printf("M=%d, N=%d, BS=%d\n",M,N,block_size);
    for(i=0;i<nb_nodes(topology);i++){
      k[i]=-1;
    }
    for(i=0;i<M;i++){
      if(proc_list[i]!=-1){
#ifdef DEBUG
	printf ("%d->%d\n",proc_list[i],nodes_id[i/block_size]);
#endif
	sigma[proc_list[i]]=nodes_id[i/block_size];
	k[nodes_id[i/block_size]]=proc_list[i];
      }
    }
  }else{
    printf("M=%d, N=%d, BS=%d\n",M,N,block_size);
    for(i=0;i<M;i++){
      if(proc_list[i]!=-1){
#ifdef DEBUG
	printf ("%d->%d\n",proc_list[i],nodes_id[i/block_size]);
#endif
	sigma[proc_list[i]]=nodes_id[i/block_size];
      }
    }

  }
  free(proc_list);

}

void map_topology_simple(tm_topology_t *topology,tree_t *comm_tree, int *sigma,int *k){
  map_topology(topology,comm_tree,topology->nb_nodes[topology->nb_levels-1],topology->nb_levels-1,sigma,k);
}

static int int_cmp(const void* x1,const void* x2){

  int *e1,*e2;

  e1=((int *)x1);
  e2=((int *)x2);

  
  return (*e1)>(*e2)?-1:1;
} 


int decompose(int n,int optimize,int *tab){
  int primes[6]={2,3,5,7,11,0},i=0;
  int flag=2;
  int j=1;


  while(primes[i]&&(n!=1)){
    //    printf("[%d] before=%d\n",primes[i],n);
    if(flag&&optimize&&(n%primes[i]!=0)){
      n+=primes[i]-n%primes[i];
      flag--;
      i=0;
      continue;
    }
    //printf("after=%d\n",n);
    if(n%primes[i]==0){
      tab[j++]=primes[i];
      n/=primes[i];
    }else{
      i++;
      flag=1;
    }
  }
  if(n!=1){
    tab[j++]=n;
  }

  qsort(tab+1,j-1,sizeof(int),int_cmp);

  for(i=0;i<j;i++)
    printf("%d:",tab[i]);
  printf("\n");

  tab[j]=0;
  
  return j+1;
}


tree_t *build_synthetic_topology_old(int *synt_tab,int id,int depth,int nb_levels){
  tree_t *res,**child;
  int arity=synt_tab[0];
  int val,i; 

  res=(tree_t*)malloc(sizeof(tree_t));
  val=0;
  if(depth>=nb_levels)
    child=NULL;
  else{
    child=(tree_t**)malloc(sizeof(tree_t*)*arity);
    for(i=0;i<arity;i++){
      child[i]=build_synthetic_topology_old(synt_tab+1,i,depth+1,nb_levels);
    child[i]->parent=res;
    val+=child[i]->val;
    }
  }
  set_node(res,child,arity,NULL,id,val+speed(depth),child[0]);
  return res;
}


void display_topology(tm_topology_t *topology){
  int i,j;
  for(i=0;i<topology->nb_levels;i++){
    printf("%d: ",i);
    for(j=0;j<topology->nb_nodes[i];j++)
      printf("%d ",topology->node_id[i][j]);
    printf("\n");
  }
}

/* 
   Build a synthetic balanced topology

   arity : array of arity of the first nb_level (of size nb_levels-1)
   core_numbering: numbering of the core by the system. Array of size nb_core_per_node

   nb_core_per_nodes: number of cores of a given node

   The numbering of the cores is done in round robin fashion after a width traversal of the topology
 */

tm_topology_t  *build_synthetic_topology(int *arity, int nb_levels, int *core_numbering, int nb_core_per_nodes){
  tm_topology_t *topology;
  int i,j,n=1;
  
  topology=(tm_topology_t*)malloc(sizeof(tm_topology_t));
  topology->arity=(int*)malloc(sizeof(int)*nb_levels);
  memcpy(topology->arity,arity,sizeof(int)*nb_levels);
  topology->nb_levels=nb_levels;

  topology->node_id=(int**)malloc(sizeof(int*)*topology->nb_levels);
  topology->nb_nodes=(int*)malloc(sizeof(int)*topology->nb_levels);


  for(i=0;i<topology->nb_levels;i++){
    topology->nb_nodes[i]=n;
    topology->node_id[i]=(int*)malloc(sizeof(int)*n);
    if(i<topology->nb_levels-1){
      for(j=0;j<n;j++)
	topology->node_id[i][j]=j;
    }else{
      for(j=0;j<n;j++)
	topology->node_id[i][j]=core_numbering[j%nb_core_per_nodes]+(nb_core_per_nodes)*(j/nb_core_per_nodes);
    }

    n*=topology->arity[i]; 
  }
  return topology;
  
}




void   build_synthetic_proc_id(tm_topology_t *topology){
  int n=1,i,j;
  topology->node_id=(int**)malloc(sizeof(int*)*topology->nb_levels);
  topology->nb_nodes=(int*)malloc(sizeof(int)*topology->nb_levels);


  for(i=0;i<topology->nb_levels;i++){
    topology->nb_nodes[i]=n;
    topology->node_id[i]=(int*)malloc(sizeof(int)*n);
    for(j=0;j<n;j++)
      topology->node_id[i][j]=j;
    n*=topology->arity[i]; 
  }
 

 
}

void update_comm_speed(double **comm_speed,int old_size,int new_size){
  double *old_tab,*new_tab;
  int i;
  printf("comm speed [%p]: ",*comm_speed);

  old_tab=*comm_speed;
  new_tab=(double*)malloc(sizeof(double)*new_size);
  *comm_speed=new_tab;

  for(i=0;i<new_size;i++){
    if(i<old_size)
      new_tab[i]=old_tab[i];
    else
      new_tab[i]=new_tab[i-1];

    printf("%f ",new_tab[i]);
  }
  
  printf("\n");
}



/* d: size of comm_speed */
void TreeMatchMapping(int nb_obj, int nb_proc, double **comm_mat,  double *obj_weight, double * comm_speed, int d, int *sol){
  tree_t *comm_tree;
  tm_topology_t *topology;
  double duration;

  int i;
  TIC;
  
  for(i=0;i<nb_obj;i++){
    sol[i]=i;
    //    printf("%f ",obj_weight[i]);
  }
  //printf("\n");
  

  //  return;

  topology=(tm_topology_t*)malloc(sizeof(tm_topology_t));
  topology->arity=(int*)malloc(sizeof(int)*MAX_LEVELS);
  topology->arity[0]=nb_proc;
  topology->nb_levels=decompose((int)ceil((1.0*nb_obj)/nb_proc),1,topology->arity);
  printf("Topology nb levels=%d\n",topology->nb_levels);
  build_synthetic_proc_id(topology);

  if(topology->nb_levels>d)
    update_comm_speed(&comm_speed,d,topology->nb_levels);

  //exit(-1);
  //topology_to_arch(topology);

  //display_tab(arch,hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PROC));
  //display_tab(arch,96);
  //exit(-1);
  //int nb_core=topo_nb_proc(topology,1000);

  //display_tab(comm_mat,N);

  TIC;
  comm_tree=build_tree_from_topology(topology,comm_mat,nb_obj,obj_weight,comm_speed);
  printf("Tree buildinbg time=%f\n",TOC);
  TIC;
  map_topology(topology,comm_tree,nb_proc,1,sol,NULL);
  printf("Topology mapping time=%f\n",TOC);


  if(topology->nb_levels>d)
    free(comm_speed);

  free_topology(topology);
  free_tree(comm_tree);

  duration=TOC;
  printf("-------------- Mapping done in %.4fs!\n",duration);
}

void display_other_heuristics(tm_topology_t *topology,int N,double **comm,double **arch){
  CLOCK_T time1,time0;
  double duration; 
  int *sol;
 
  sol=(int*)malloc(sizeof(int)*N);

  map_Packed(topology,N,sol);
  printf("Packed: "); 
  print_sol(N,sol,comm,arch);



  map_RR(N,sol);
  printf("RR: "); 
  print_sol(N,sol,comm,arch);

  CLOCK(time0);
  map_MPIPP(topology,1,N,sol,comm,arch);
  CLOCK(time1);
  duration=CLOCK_DIFF(time1,time0);
  printf("MPIPP-1-D:%f\n",duration);
  printf("MPIPP-1: ");
  print_sol(N,sol,comm,arch);

  CLOCK(time0);
  map_MPIPP(topology,5,N,sol,comm,arch);
  CLOCK(time1);
  duration=CLOCK_DIFF(time1,time0);
  printf("MPIPP-5-D:%f\n",duration);
  printf("MPIPP-5: ");
  print_sol(N,sol,comm,arch);

  free(sol);
}


