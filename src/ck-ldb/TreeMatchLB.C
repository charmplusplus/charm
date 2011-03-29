/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <charm++.h>
extern "C"{
#include "tm_tree.h"
};
#include "TreeMatchLB.h"

CreateLBFunc_Def(TreeMatchLB, "TreeMatch load balancer, like a normal one but with empty strategy")

#include "TreeMatchLB.def.h"

TreeMatchLB::TreeMatchLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = (char*)"TreeMatchLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] TreeMatchLB created\n",CkMyPe());
}

CmiBool TreeMatchLB::QueryBalanceNow(int _step)
{
  return CmiTrue;
}


double *get_comm_speed(int *depth){
  double *res;
  int i;
  
  *depth=5;

  if(! (*depth))
    return NULL;
  
  res=(double*)malloc(sizeof(double)*(*depth));
  res[0]=1;

  for(i=1;i<*depth;i++){
    res[i]=res[i-1]*res[i-1];
  }
}

void TreeMatchLB::work(BaseLB::LDStats* stats)
{
  int nb_obj,nb_proc;
  double **comm_mat;
  int i;
  int *permut_vec;
  int count = stats->nprocs();
  double *obj_weight;

  nb_proc=count;
  nb_obj=stats->n_objs;
  
  stats->makeCommHash();
  // allocate object weight matrix
  obj_weight=(double*)malloc(sizeof(double)*nb_obj);
  // allocate communication matrix
  comm_mat=(double**)malloc(sizeof(double*)*nb_obj);
  for(i=0;i<nb_obj;i++){
    comm_mat[i]=(double*)calloc(nb_obj,sizeof(double));
  }

  for(i=0;i<stats->n_comm;i++){
    LDCommData &commData = stats->commData[i];
    if((!commData.from_proc())&&(commData.recv_type()==LD_OBJ_MSG)){
      int from = stats->getHash(commData.sender);
      int to = stats->getHash(commData.receiver.get_destObj());
      comm_mat[from][to]+=commData.bytes;
      comm_mat[to][from]+=commData.bytes;
    }
  }

  for(i=0;i<n_objs;i++){
    LDObjData &oData = stats->objData[i];
    obj_weight[i] = oData.wallTime ;
  }
  
  int depthxs;
  comm_speed=get_comm_speed(&depth);
  TreeMatchMapping(nb_obj, nb_proc, comm_mat, obj_weight, comm_speed, depth, stats->to_proc.getVec());
  

  // free communication matrix;
  for(i=0;i<nb_obj;i++){
      free(comm_mat[i]);
  }
  free(comm_mat);
  free(comm_speed);
  free(obj_weight);
}

/*@}*/
