#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "tm_tree.h"
#include "tm_timings.h"
#include "tm_bucket.h"

#if __CHARMC__
#include "converse.h"
#else
#define CmiLog2 log2
#endif

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#undef DEBUG


void free_list_child(tree_t *tree){
  int i;
  if(tree){
    for(i=0;i<tree->arity;i++){
      free_list_child(tree->child[i]);
    }
    free(tree->child);
    if(tree->dumb)
      free(tree);
  }
}




void free_tab_child(tree_t *tree){
  if(tree){
    free_tab_child(tree->tab_child);
    free(tree->tab_child);
  }
}



void free_tree(tree_t *tree){

  free_list_child(tree);
  free_tab_child(tree);
  free(tree);
}

long int choose (long n,long k){
  // compute C_n_k
  double res=1;
  int i;
  for(i=0;i<k;i++)
    res*=(double)(n-i)/(double)(k-i);
  
  return (long int)res;
}

void set_node(tree_t *node,tree_t ** child, int arity,tree_t *parent,int id,double val,tree_t *tab_child){
  static int uniq=0;
  node->child=child;
  node->arity=arity;
  node->tab_child=NULL;
  node->parent=parent;
  node->id=id;
  node->val=val;
  node->uniq=uniq++;
  node->dumb=0;
}

void display_node(tree_t *node){
  printf("child : %p\narity : %d\nparent : %p\nid : %d\nval : %f\nuniq : %d\n\n" ,
	 node->child,node->arity,node->parent,node->id,node->val,node->uniq);
}
void clone_tree(tree_t *new,tree_t *old){
  int i;
  new->child=old->child;
  new->arity=old->arity;
  new->parent=old->parent;
  //new->deb_tab_child=old->deb_tab_child;
  new->id=old->id;
  new->val=old->val;
  new->uniq=old->uniq;
  new->dumb=old->dumb;
  for(i=0;i<new->arity;i++)
    new->child[i]->parent=new;
}



double *aggregate_obj_weight(tree_t *new_tab_node, double *tab, int M){
  int i,i1,id1;
  double *res;
  
  if(!tab)
    return NULL;
  
  res=(double*)malloc(M*sizeof(double));
  
  for(i=0;i<M;i++){
    res[i]=0.0;
    for(i1=0;i1<new_tab_node[i].arity;i1++){
      id1=new_tab_node[i].child[i1]->id;
      res[i]+=tab[id1];
    }
  }
  
  return res;

}


double **aggregate_com_mat(tree_t *new_tab_node, double **tab, int M){
  int i,j,i1,j1,id1,id2;
  double **res;
    
  

  res=(double**)malloc(M*sizeof(double*));
  for(i=0;i<M;i++)
    res[i]=(double*)malloc(M*sizeof(double));
  
  for(i=0;i<M;i++){
    for(j=0;j<M;j++){
      if(i==j)
	res[i][j]=0;
      else{
	res[i][j]=0;
	for(i1=0;i1<new_tab_node[i].arity;i1++){
	  id1=new_tab_node[i].child[i1]->id;
	  for(j1=0;j1<new_tab_node[j].arity;j1++){
	    id2=new_tab_node[j].child[j1]->id;
	    res[i][j]+=tab[id1][id2];
	    //printf("res[%d][%d]+=tab[%d][%d]=%f\n",i,j,id1,id2,tab[id1][id2]);
	  }
	}
      }
    }
  }
  
  return res;

}

void free_tab_double(double**tab,int N){
  int i;
  for(i=0;i<N;i++){
    free(tab[i]);
  }
  free(tab);
}

void free_tab_int(int**tab,int N){
  int i;
  for(i=0;i<N;i++){
    free(tab[i]);
  }
  free(tab);
}

void display_tab(double **tab,int N){
  int i,j;
  double line,total=0;;
  for(i=0;i<N;i++){
    line=0;
    for(j=0;j<N;j++){
      printf("%g ",tab[i][j]);
      line+=tab[i][j];
    }
    total+=line;
    //printf(": %g",line);
    printf("\n");
  }
  printf("Total: %.2f\n",total);
}


double eval_grouping(double **tab,tree_t **cur_group,int arity,int N){
  double res=0;
  int i,k,j,id,id1,id2;
  //display_tab(tab,N);


  for(i=0;i<arity;i++){
    id=cur_group[i]->id;
    //printf("%d ",id);
    for(k=0;k<N;k++){
      //printf("res+=tab[%d][%d]+tab[%d][%d]=%f+%f \n",k,id,id,k,tab[k][id],tab[id][k]);
      res+=tab[k][id];
    }
  }
  
  for(i=0;i<arity;i++){
    id1=cur_group[i]->id;
    for(j=0;j<arity;j++){
      id2=cur_group[j]->id;
      //printf("res-=tab[%d][%d]=%f\n",id1,id2,tab[id1][id2]);
      res-=tab[id1][id2];
    }
  }
  //printf(" = %f\n",res);
  return res;
}



group_list_t *new_group_list(tree_t **tab,double val,group_list_t *next){
  group_list_t *res;
  res=(group_list_t *)malloc(sizeof(group_list_t));
  res->tab=tab;
  res->val=val;
  res->next=next;
  res->sum_neighbour=0;
  return res;
}


void add_to_list(group_list_t *list,tree_t **cur_group, int arity, double val){

  group_list_t *elem;
  int i;
  tree_t **tab;
  
  tab=(tree_t **)malloc(sizeof(tree_t *)*arity);

  for(i=0;i<arity;i++){
    tab[i]=cur_group[i];
    //printf("%d ",cur_group[i]->id);
  }

  //printf("\n");
  elem=new_group_list(tab,val,list->next);
  list->next=elem;
  list->val++;
  /* cppcheck-suppress memleak */
}



void  list_all_possible_groups(double **tab,tree_t *tab_node,int id,int arity, int depth,
			       tree_t **cur_group,int N,group_list_t *list){
  double val;
  int i;



  if(depth==arity){
    val=eval_grouping(tab,cur_group,arity,N);
    add_to_list(list,cur_group,arity,val);
    return;
    }else if(N+depth>=arity+id){
    //}else if(1){
    for(i=id;i<N;i++){
      if(tab_node[i].parent)
	continue;
      cur_group[depth]=&tab_node[i];
      //printf("%d<-%d\n",depth,i);
      list_all_possible_groups(tab,tab_node,i+1,arity,depth+1,cur_group,N,list);
    }
  }
}


void update_val(double **tab,tree_t *parent,int N){
  int i;

  parent->val=eval_grouping(tab,parent->child,parent->arity,N);
  //printf("connecting: ");
  for(i=0;i<parent->arity;i++){
    //printf("%d ",parent->child[i]->id);
    /*  if(parent->child[i]->parent!=parent){
      parent->child[i]->parent=parent;
    }else{
      fprintf(stderr,"redundant operation!\n");
      exit(-1);
      }*/
  }


  //printf(": %f\n",parent->val);
}


int independent_groups(group_list_t **selection,int d,group_list_t *elem,int arity){
  int i,j,k;

  if(d==0)
    return 1;

  for(i=0;i<arity;i++){
    for(j=0;j<d;j++){
      for(k=0;k<arity;k++){
	if(elem->tab[i]->id==selection[j]->tab[k]->id)
	  return 0;
      }
    }
  }
  
  return 1;


}

void display_selection (group_list_t** selection,int M,int arity,double val){
  int i,j;
  for(i=0;i<M;i++){
    for(j=0;j<arity;j++)
      printf("%d ",selection[i]->tab[j]->id);
    printf("-- ");
  }
  printf(":%f\n",val); 

}

void display_grouping (tree_t *father,int M,int arity,double val){
  int i,j;
  for(i=0;i<M;i++){
    for(j=0;j<arity;j++)
      printf("%d ",father[i].child[j]->id);
    printf("-- ");
  }
  printf(":%f\n",val); 

}


int recurs_select_independent_groups(group_list_t **tab,int i,int n,int arity,int d,int M,double val,double *best_val,group_list_t **selection,group_list_t **best_selection){
  group_list_t *elem;

  //if(val>=*best_val)
  // return 0;

  if(d==M){
    //display_selection(selection,M,arity,val);
    if(val<*best_val){
      *best_val=val;
      for(i=0;i<M;i++)
	best_selection[i]=selection[i];
      return 1;
    }
    return 0;
  }
  
  while(i<n){
    elem=tab[i];
    if(independent_groups(selection,d,elem,arity)){
      //printf("%d: %d\n",d,i);
      selection[d]=elem;
      val+=elem->val;
      return recurs_select_independent_groups(tab,i+1,n,arity,d+1,M,val,best_val,selection,best_selection);
    }
    i++;
  }
  return 0;
}


int test_independent_groups(group_list_t **tab,int i,int n,int arity,int d,int M,double val,double *best_val,group_list_t **selection,group_list_t **best_selection){
  group_list_t *elem;

  if(d==M){
    //display_selection(selection,M,arity,val);
    return 1;
  }
  
  while(i<n){
    elem=tab[i];
    if(independent_groups(selection,d,elem,arity)){
      //printf("%d: %d\n",d,i);
      selection[d]=elem;
      val+=elem->val;
      return recurs_select_independent_groups(tab,i+1,n,arity,d+1,M,val,best_val,selection,best_selection);
    }
    i++;
  }
  return 0;
}

void  delete_group_list(group_list_t *list){

  if(list){
    delete_group_list(list->next);
    free(list->tab);
    free(list);
  }
}


int group_list_id(const void* x1,const void* x2){ 

  group_list_t *e1,*e2;

  e1=*((group_list_t**)x1);
  e2=*((group_list_t**)x2);

  
  return e1->tab[0]->id<e2->tab[0]->id?-1:1;
} 

int group_list_asc(const void* x1,const void* x2){ 

  group_list_t *e1,*e2;

  e1=*((group_list_t**)x1);
  e2=*((group_list_t**)x2);

  
  return e1->val<e2->val?-1:1;
} 


int group_list_dsc(const void* x1,const void* x2){ 

  group_list_t *e1,*e2;

  e1=*((group_list_t**)x1);
  e2=*((group_list_t**)x2);

  
  return e1->val>e2->val?-1:1;
} 


int weighted_degree_asc(const void* x1,const void* x2){ 

  group_list_t *e1,*e2;

  e1=*((group_list_t**)x1);
  e2=*((group_list_t**)x2);

  
  return e1->wg>e2->wg?1:-1;
} 


int weighted_degree_dsc(const void* x1,const void* x2){ 

  group_list_t *e1,*e2;

  e1=*((group_list_t**)x1);
  e2=*((group_list_t**)x2);

  
  return e1->wg>e2->wg?-1:1;
} 

int  select_independent_groups(group_list_t **tab_group,int n,int arity,int M,double *best_val,group_list_t **best_selection,int bound,double max_duration){
  int i;
  group_list_t **selection;
  double val,duration;
  CLOCK_T time1,time0;


  selection=(group_list_t **)malloc(sizeof(group_list_t*)*M);
  CLOCK(time0);
  for(i=0;i<MIN(bound,n);i++){
    selection[0]=tab_group[i];
    val=tab_group[i]->val;
    recurs_select_independent_groups(tab_group,i+1,n,arity,1,M,val,best_val,selection,best_selection);
    if(i%5){
      CLOCK(time1);
      duration=CLOCK_DIFF(time1,time0);
      if(duration>max_duration){
	free(selection);
	return 1;
      }
    }
  }

  free(selection);

#ifdef DEBUG
  display_selection(best_selection,M,arity,*best_val);
#endif
  return 0;
}

int  select_independent_groups_by_largest_index(group_list_t **tab_group,int n,int arity,int M,double *best_val,group_list_t **best_selection,int bound,double max_duration){
  int i,nb_groups=0;
  group_list_t **selection;
  double val,duration;
  int dec;
  CLOCK_T time1,time0;

  selection=(group_list_t **)malloc(sizeof(group_list_t*)*M);
  CLOCK(time0);
  
  dec=MAX(n/10000,1);
  for(i=n-1;i>=0;i-=dec*dec){
    selection[0]=tab_group[i];
    val=tab_group[i]->val;
    nb_groups+=test_independent_groups(tab_group,i+1,n,arity,1,M,val,best_val,selection,best_selection);
    //printf("%d:%d\n",i,nb_groups);
    if(nb_groups>=bound){
      free(selection);
      return 0;
    }
    if(i%5){
      CLOCK(time1);
      duration=CLOCK_DIFF(time1,time0);
      if(duration>max_duration){
	free(selection);
	return 1;
      }
    }
  }

  free(selection);
  return 0;
}

void list_to_tab(group_list_t *list,group_list_t **tab,int n){
  int i;
  for(i=0;i<n;i++){
    if(!list){
      fprintf(stderr,"Error not enough elements. Only %d on %d\n",i,n);
      exit(-1);
    }
    tab[n-i-1]=list;
    list=list->next;
  }
  if(list){
    fprintf(stderr,"Error too many elements\n");
    exit(-1);
  }
}

void display_tab_group(group_list_t **tab, int n,int arity){
  int i,j;
  for(i=0;i<n;i++){
    for(j=0;j<arity;j++)
      printf("%d ",tab[i]->tab[j]->id);
    printf(": %.2f %.2f\n",tab[i]->val,tab[i]->wg);
  }
}

int independent_tab(tree_t **tab1,tree_t **tab2,int n){
  int i,j;
  i=0;j=0;
  while((i<n)&&(j<n)){
    if(tab1[i]->id==tab2[j]->id)
      return 0;
    else if(tab1[i]->id>tab2[j]->id)
      j++;
    else
      i++;
  }
  return 1;
}

    

void compute_weighted_degree(group_list_t **tab, int n,int arity){
  int i,j;
  for(i=0;i<n;i++)
    tab[i]->sum_neighbour=0;
  for(i=0;i<n;i++){
    //printf("%d/%d=%f%%\n",i,n,(100.0*i)/n);
    for(j=i+1;j<n;j++){
      //if(!independent_groups(&tab[i],1,tab[j],arity)){
	if(!independent_tab(tab[i]->tab,tab[j]->tab,arity)){
	tab[i]->sum_neighbour+=tab[j]->val;
	tab[j]->sum_neighbour+=tab[i]->val;
     }
    }

    tab[i]->wg=tab[i]->sum_neighbour/tab[i]->val;
    if(tab[i]->sum_neighbour==0)
      tab[i]->wg=0;
    //printf("%d:%f/%f=%f\n",i,tab[i]->sum_neighbour,tab[i]->val,tab[i]->wg);
  }
}


/*
  Very slow: explore all possibilities
  tab: comm_matrix at the considered level (used to evaluate a grouping)
  tab_node: array of the node to group
  parent: node to which attached the computed group
  id: current considered node of tab_node
  arity: number of children of parent (i.e.) size of the group to compute
  best_val: current value of th grouping
  cur_group: current grouping
 */
void  group(double **tab,tree_t *tab_node,tree_t *parent,int id,int arity, int n,double *best_val,tree_t **cur_group,int N){
  double val;
  int i;


  //if we have found enough noide in the group
  if(n==arity){
    // evaluate this group
    val=eval_grouping(tab,cur_group,arity,N);
    // If we improve compared to previous grouping: uodate the children of parent accordingly
    if(val<*best_val){
      *best_val=val;
	for(i=0;i<arity;i++){
	  parent->child[i]=cur_group[i];
	}
      parent->arity=arity;
    } 
    return;
  }
  


  // If we need more node in the group
  // Continue to explore avilable nodes
  for(i=id+1;i<N;i++){
    // If this node is allready in a group: skip it
    if(tab_node[i].parent)
      continue;
    //Otherwise, add it to the group at place n
    cur_group[n]=&tab_node[i];
    //printf("%d<-%d\n",n,i);
    //recursively add the next element to this group
    group(tab,tab_node,parent,i,arity,n+1,best_val,cur_group,N);
  }
}

/*
   tab: comm_matrix at the considered level (use to evaluate a grouping)
  tab_node: array of the node to group
  parent: node to which attached the computed group
  id: current considered node of tab_node
  arity: number of children of parent (i.e.) size of the group to compute
  best_val: current value of th grouping
  cur_group: current grouping
  N: size of tab and tab_node. i.e. number of nodes at the considered level
 */
void  fast_group(double **tab,tree_t *tab_node,tree_t *parent,int id,int arity, int n,
		 double *best_val,tree_t **cur_group,int N, int *nb_groups,int max_groups){
  double val;
  int i;

  //printf("Max groups=%d\n",max_groups);

  //if we have found enough node in the group
  if(n==arity){
    (*nb_groups)++;
    // evaluate this group
    val=eval_grouping(tab,cur_group,arity,N);
    // If we improve compared to previous grouping: uodate the children of parent accordingly
    if(val<*best_val){
      *best_val=val;
	for(i=0;i<arity;i++){
	  parent->child[i]=cur_group[i];
	}
      parent->arity=arity;
    } 
    return;
  }
  


  // If we need more node in the group
  // Continue to explore avilable nodes
  for(i=id+1;i<N;i++){
    // If this node is allready in a group: skip it
    if(tab_node[i].parent)
      continue;
    //Otherwise, add it to the group at place n
    cur_group[n]=&tab_node[i];
    //printf("%d<-%d %d/%d\n",n,i,*nb_groups,max_groups);
    //exit(-1);
    //recursively add the next element to this group
    fast_group(tab,tab_node,parent,i,arity,n+1,best_val,cur_group,N,nb_groups,max_groups);
    if(*nb_groups>max_groups)
      return;
  }
}




void fast_grouping(double **tab,tree_t *tab_node, tree_t *new_tab_node, int arity,int N, int M,long int k){
  tree_t **cur_group;
  int l,i;
  double best_val,val=0;
  int nb_groups;

  cur_group=(tree_t**)malloc(sizeof(tree_t*)*arity);
  for(l=0;l<M;l++){
    best_val=DBL_MAX;
    nb_groups=0;
    //printf("k%d/%d, k=%ld\n",l,M,k);
    /* select the best greedy grouping among the 10 first one*/
    //fast_group(tab,tab_node,&new_tab_node[l],-1,arity,0,&best_val,cur_group,N,&nb_groups,MAX(2,(int)(50-log2(k))-M/10));
    fast_group(tab,tab_node,&new_tab_node[l],-1,arity,0,&best_val,cur_group,N,&nb_groups,MAX(1,(int)(50-CmiLog2(k))-M/10));
    val+=best_val;
    for(i=0;i<new_tab_node[l].arity;i++){
      new_tab_node[l].child[i]->parent=&new_tab_node[l];
    }
    update_val(tab,&new_tab_node[l],N);
  }

  free(cur_group);  

  printf("val=%f\n",val);
  //exit(-1);

#ifdef DEBUG
  display_grouping(new_tab_node,M,arity,val);
#endif
}




int adjacency_asc(const void* x1,const void* x2){ 

  adjacency_t *e1,*e2;

  e1=((adjacency_t*)x1);
  e2=((adjacency_t*)x2);

  
  return e1->val<e2->val?-1:1;
} 


int adjacency_dsc(const void* x1,const void* x2){ 

  adjacency_t *e1,*e2;

  e1=((adjacency_t*)x1);
  e2=((adjacency_t*)x2);

  
  return e1->val>e2->val?-1:1;
} 
void super_fast_grouping(double **tab,tree_t *tab_node, tree_t *new_tab_node, int arity,int N, int M,int k){
  double val=0;
  adjacency_t *graph;
  int i,j,e,l,nb_groups;
  double duration;

  assert(arity==2);

  TIC;
  graph=(adjacency_t*)malloc(sizeof(adjacency_t)*((N*N-N)/2));
  e=0;
  for(i=0;i<N;i++){
    for(j=i+1;j<N;j++){
      graph[e].i=i;
      graph[e].j=j;
      graph[e].val=tab[i][j];
      e++;
    }
  }
  duration=TOC;
  printf("linearization=%fs\n",duration);
  

  assert(e==(N*N-N)/2);
  TIC;  
  qsort(graph,e,sizeof(adjacency_t),adjacency_dsc);
  duration=TOC;
  
  printf("sorting=%fs\n",duration);

  TIC;

TIC;
  l=0;
  nb_groups=0;
  for(i=0;i<e&&l<M;i++){
    if(try_add_edge(tab,tab_node,&new_tab_node[l],arity,graph[i].i,graph[i].j,N,&nb_groups)){
      l++;
    }
  }

  for(l=0;l<M;l++){
    update_val(tab,&new_tab_node[l],N);      
    val+=new_tab_node[l].val;
  }

  duration=TOC;
  printf("Grouping=%fs\n",duration);

  printf("val=%f\n",val);

  free(graph);

#ifdef DEBUG
  display_grouping(new_tab_node,M,arity,val);
#endif
}



double **build_cost_matrix(double **comm_matrix, double* obj_weight, double comm_speed, int N){
  double **res,avg;
  int i,j;

  if(!obj_weight)
    return comm_matrix;

  res=(double**)malloc(N*sizeof(double*));
  for(i=0;i<N;i++)
    res[i]=(double*)malloc(N*sizeof(double));
  
  avg=0;
  for(i=0;i<N;i++)
    avg+=obj_weight[i];
  avg/=N;

  printf("avg=%f\n",avg);

  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      if(i==j)
	res[i][j]=0;
      else
	res[i][j]=1e-4*comm_matrix[i][j]/comm_speed-fabs(avg-(obj_weight[i]+obj_weight[j])/2);

  return res;
  
}



/*
  comm_matrix: comm_matrix at the considered level (use to evaluate a grouping) of size N*N
  tab_node: array of the node to group
  new_tab_node: array of nodes at the next level (the parents of the node in tab_node once the grouping will be done). 
  arity: number of children of parent (i.e.) size of the group to compute
  N: size of tab and tab_node. i.e. number of nodes at the considered level
  M: size of new_tab_node (i.e) the number of parents
  Hence we have: M*arity=N
*/void group_nodes(double **comm_matrix,tree_t *tab_node, tree_t *new_tab_node, int depth, int arity,int N, int M, double* obj_weigth, double comm_speed){
  tree_t **cur_group;
  int j,l,n;
  long int k;
  group_list_t list,**best_selection,**tab_group;
  double best_val,last_best;
  int timeout;
  double **tab; /*cost matrix taking into account the communiocation cost but also the weight of the object*/
  double duration;

  TIC;

  /* might return comm_matrix (if obj_weight==NULL): do not free this tab in this case*/ 
  tab=build_cost_matrix(comm_matrix,obj_weigth,comm_speed,N);



  k=choose(N,arity);
  printf("Number of groups:%ld\n",k);
  
  if(k>30000||depth>5){
  #ifdef DEBUG
    printf("Fast Grouping...\n");
#endif

    double duration;
    
    TIC;
    if(arity<=2){
      //super_fast_grouping(tab,tab_node,new_tab_node,arity,N,M,k);
      bucket_grouping(tab,tab_node,new_tab_node,arity,N,M,k);
    }else{
      fast_grouping(tab,tab_node,new_tab_node,arity,N,M,k);
    }
    duration=TOC;

    printf("Fast grouping duration=%f\n",duration);
    
    //exit(-1);
  }else{
#ifdef DEBUG
    printf("Grouping nodes...\n");
#endif
    list.next=NULL;
    list.val=0;//number of elements in the list
    cur_group=(tree_t**)malloc(sizeof(tree_t*)*arity);
    best_selection=(group_list_t **)malloc(sizeof(group_list_t*)*M);
  
    list_all_possible_groups(tab,tab_node,0,arity,0,cur_group,N,&list);
    n=(int)list.val;
    assert(n==k);
    tab_group=(group_list_t**)malloc(sizeof(group_list_t)*n);
    list_to_tab(list.next,tab_group,n);
#ifdef DEBUG
    printf("List to tab done\n");
#endif
    best_val=DBL_MAX;
  
    // perform the pack mapping fist
    timeout=select_independent_groups(tab_group,n,arity,M,&best_val,best_selection,1,0.1);
#ifdef DEBUG
    if(timeout){
      printf("Packed mapping timeout!\n");
    }
#endif
    // give this mapping an exra credit (in general MPI application are made such that neighbour process communicates more than distant ones)
    best_val/=1.001;
#ifdef DEBUG
    printf("Packing computed\n");
#endif
    // perform a mapping trying to use group that cost less first
    qsort(tab_group,n,sizeof(group_list_t*),group_list_asc);
    last_best=best_val;
    timeout=select_independent_groups(tab_group,n,arity,M,&best_val,best_selection,10,0.1);
#ifdef DEBUG
    if(timeout){
      printf("Cost less first timeout!\n");
    }else if(last_best>best_val){
      printf("Cost less first Impoved solution\n");
    }
    printf("----\n");
#endif
    // perform a mapping trying to minimize the use of groups that cost a lot 
    qsort(tab_group,n,sizeof(group_list_t*),group_list_dsc);
    last_best=best_val;
    timeout=select_independent_groups_by_largest_index(tab_group,n,arity,M,&best_val,best_selection,10,0.1);
#ifdef DEBUG
    if(timeout){
      printf("Cost most last timeout!\n");
    }else if(last_best>best_val){
      printf("Cost most last impoved solution\n");
    }
#endif
    if(n<10000){
      // perform a mapping in the weighted degree order 
#ifdef DEBUG
      printf("----WG----\n");
#endif
      compute_weighted_degree(tab_group,n,arity);
#ifdef DEBUG
      printf("Weigted degree computed\n");
#endif
      qsort(tab_group,n,sizeof(group_list_t*),weighted_degree_dsc);
      //display_tab_group(tab_group,n,arity);
      last_best=best_val;
      timeout=select_independent_groups(tab_group,n,arity,M,&best_val,best_selection,10,0.1);
#ifdef DEBUG
      if(timeout){
	printf("WG timeout!\n");
      }else if(last_best>best_val){
	printf("WG impoved solution\n");
      }
#endif
    }
    
    qsort(best_selection,M,sizeof(group_list_t*),group_list_id);

    for(l=0;l<M;l++){
      for(j=0;j<arity;j++){
	new_tab_node[l].child[j]=best_selection[l]->tab[j];
	new_tab_node[l].child[j]->parent=&new_tab_node[l];
      }
      new_tab_node[l].arity=arity;
    
      //printf("arity=%d\n",new_tab_node[l].arity);
      update_val(tab,&new_tab_node[l],N);
    }
    
    delete_group_list((&list)->next);
    free(best_selection);
    free(tab_group);
    free(cur_group);  
  }

  if(tab!=comm_matrix)
    free_tab_double(tab,N);
  
  duration=TOC;
  printf("Grouping done in %.4fs!\n",duration);
}





void complete_com_mat(double ***tab,int N, int K){
  double **old_tab,**new_tab;
  int M,i,j;

  old_tab=*tab;
  
  M=N+K;
  new_tab=(double**)malloc(M*sizeof(double*));
  for(i=0;i<M;i++)
    new_tab[i]=(double*)malloc(M*sizeof(double));
  
  *tab=new_tab;
  for(i=0;i<M;i++){
    for(j=0;j<M;j++){
      if((i<N)&&(j<N)){
	new_tab[i][j]=old_tab[i][j];
      }else{
	new_tab[i][j]=0;
      }
    }
  }
}

void complete_obj_weight(double **tab,int N, int K){
  double *old_tab,*new_tab,avg;
  int M,i;

  old_tab=*tab;

  if(!old_tab)
    return;

  
  avg=0;
  for(i=0;i<N;i++)
    avg+=old_tab[i];
  avg/=N;

  
  M=N+K;
  new_tab=(double*)malloc(M*sizeof(double));
  
  *tab=new_tab;
  for(i=0;i<M;i++){
    if(i<N){
      new_tab[i]=old_tab[i];
    }else{
      new_tab[i]=avg;
    }
  }
}



void create_dumb_tree(tree_t *node,int depth,tm_topology_t *topology){
  tree_t **list_child;
  int arity,i;

  
  if(depth==topology->nb_levels-1){
    set_node(node,NULL,0,NULL,-1,0,NULL);  
    return;
  }

  arity=topology->arity[depth];
  assert(arity>0);
  list_child=(tree_t**)calloc(arity,sizeof(tree_t*));
  for(i=0;i<arity;i++){
    list_child[i]=(tree_t*)malloc(sizeof(tree_t));
    create_dumb_tree(list_child[i],depth+1,topology);
    list_child[i]->parent=node;
    list_child[i]->dumb=1;
  }

  set_node(node,list_child,arity,NULL,-1,0,list_child[0]);
  
}

void complete_tab_node(tree_t **tab,int N, int K,int depth,tm_topology_t *topology){
  tree_t *old_tab,*new_tab;
  int M,i;
  if(K==0)
    return;

  old_tab=*tab;

  
  M=N+K;
  new_tab=(tree_t*)malloc(M*sizeof(tree_t));
  
  *tab=new_tab;
  for(i=0;i<M;i++){
    if((i<N)){
      clone_tree(&new_tab[i],&old_tab[i]);
    }else{
      create_dumb_tree(&new_tab[i],depth,topology);
      new_tab[i].id=i;
    }
  }

  //do not suppress tab if you are at the depth-most level it will be used at the mapping stage
  free(old_tab);  
}


void set_deb_tab_child(tree_t *tree, tree_t *child,int depth){
  //printf("depth=%d\t%p\t%p\n",depth,child,tree);
  if(depth>0)
    set_deb_tab_child(tree->tab_child,child,depth-1);
  else
    tree->tab_child=child;
}



/* 
Build the tree of the matching. It is a bottom up algorithm: it starts from the bottom of the tree on proceed by decreasing the depth 
It groups nodes of the matrix tab and link these groups to the nodes of the under level. 
Then it calls recursivcely the function to prefrom the grouping at the above level. 

tab_node: array of nodes of the under level. 
tab: local communication matrix 
N: number of nodes. Order of com_mat, size of obj_weight 
arity: arity of the nodes of the above level.
depth: current depth of the algorithm
toplogy: description of the hardware topology.  
*/
tree_t *build_level_topology(tree_t *tab_node,double **com_mat,int N,int arity,int depth,tm_topology_t *topology, double *obj_weight, double *comm_speed){
  int M; /*N/Arity: number the groups*/
  int K=0,i;
  tree_t *new_tab_node; /*array of node for this level (of size M): there will be linked to the nodes of tab_nodes*/
  double **new_com_mat; /*New communication matrix (after grouyping nodes together)*/
  tree_t *res; /*resulting tree*/
  int completed=0;
  double speed; /* communication speed at this level*/
  double *new_obj_weight;
  if((depth==0)){
    if((N==1)&&(depth==0))
      return &tab_node[0];
    else{
      fprintf(stderr,"Error: matrix size: %d and depth:%d (should be 1 and -1 respectively)\n",N,depth);
      exit(-1);
    }
  }

  
  /* If the number of nodes does not devide the arity: we add K nodes  */
  if(N%arity!=0){
    K=arity*((N/arity)+1)-N;
    //printf("****N=%d arity=%d K=%d\n",N,arity,K);  
    //display_tab(tab,N);
    /* add K rows and columns to comm_matrix*/
    complete_com_mat(&com_mat,N,K);
    /* add K element to the object weight*/
    complete_obj_weight(&obj_weight,N,K);
    //display_tab(tab,N+K);
    /* add a dumb tree to the K new "virtual nodes"*/
    complete_tab_node(&tab_node,N,K,depth,topology);
    completed=1; /*flag this addition*/
    N+=K; /*increase the number of nodes accordingly*/
  } //display_tab(tab,N);


  M=N/arity;
  printf("Depth=%d\tnb_nodes=%d\tnb_groups=%d\tsize of groups(arity)=%d\n",depth,N,M,arity);

  /*create the new nodes*/
  new_tab_node=(tree_t*)malloc(sizeof(tree_t)*M);
  /*intitialize each node*/
  for(i=0;i<M;i++){
    tree_t **list_child;
    list_child=(tree_t**)calloc(arity,sizeof(tree_t*));
    set_node(&new_tab_node[i],list_child,arity,NULL,i,0,tab_node);
  }

  /*Core of the algorithm: perfrom the grouping*/
  if(comm_speed)
    speed=comm_speed[depth];
  else
    speed=-1;
  group_nodes(com_mat,tab_node,new_tab_node,depth,arity,N,M,obj_weight,speed);
 
  /*based on that grouping aggregate the communication matrix*/
  new_com_mat=aggregate_com_mat(new_tab_node,com_mat,M);
  /*based on that grouping aggregate the object weight matrix*/
  new_obj_weight=aggregate_obj_weight(new_tab_node,obj_weight,M);

  /* set ID of virtual nodes to -1*/
  for(i=N-K;i<N;i++)
    tab_node[i].id=-1;

  //for(i=0;i<N;i++)
  //  display_node(&tab_node[i]);

  //display_tab(new_com_mat,M);

  /* decrease depth and compute arity of the above level*/
  depth--;
  if(depth>0)
    arity = topology->arity[depth-1];
  else
    arity=1;
  // assume all objects have the same arity
  res = build_level_topology(new_tab_node,new_com_mat,M,arity,depth,topology,new_obj_weight,comm_speed);  

  set_deb_tab_child(res,tab_node,depth);


  if(completed){
    free_tab_double(com_mat,N);
    free(obj_weight);
  }
  /* cppcheck-suppress deallocDealloc */
  free_tab_double(new_com_mat,M);
  free(new_obj_weight);

  return res;
}



double speed(int depth){
  // Bertha values
  //double tab[5]={21,9,4.5,2.5,0.001};
  //double tab[5]={1,1,1,1,1};
  //double tab[6]={100000,10000,1000,500,100,10};
  double tab[5]={100000,10000,1000,500,10};
    
  return 1.0/tab[depth];
  //return 10*log(depth+2);
  //return (depth+1);
  //return (long int)pow(100,depth);
}

tree_t * build_tree_from_topology(tm_topology_t *topology,double **tab,int N,double *obj_weight, double *comm_speed){
  int depth,i; 
  tree_t *res,*tab_node;


  tab_node=(tree_t*)malloc(sizeof(tree_t)*N);
  for(i=0;i<N;i++)
    set_node(&tab_node[i],NULL,0,NULL,i,0,NULL); 
 

  depth = topology->nb_levels -1;
  printf("nb_levels=%d\n",depth+1);
  // assume all objects have the same arity
  res = build_level_topology(tab_node,tab,N,topology->arity[depth-1],depth,topology, obj_weight, comm_speed);
  printf("Build tree done!\n");
  return res;
}
