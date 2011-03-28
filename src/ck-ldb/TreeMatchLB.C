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
#include "treemapping.h"
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


void TreeMatchLB::work(BaseLB::LDStats* stats)
{
  int nb_obj,nb_proc;
  double **comm_mat;
  int i;
  int *permut_vec;
  int count = stats->nprocs();

  nb_proc=count;
  nb_obj=stats->n_objs;
  
  stats->makeCommHash();
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
      comm_mat[from][to]=commData.bytes;
      comm_mat[to][from]=commData.bytes;
    }
  }
  
  TreeMatchMapping(nb_obj,nb_proc,comm_mat,stats->to_proc.getVec());

  // free communication matrix;
  for(i=0;i<nb_obj;i++){
      free(comm_mat[i]);
  }
  free(comm_mat);
}

/*@}*/
