#ifndef __DECISIONTREE__H__
#define __DECISIONTREE__H__

#include <vector>
#include "picstreenode.h"

class DecisionTree {

  TreeNode *root;

public:

  DecisionTree() ;
  DecisionTree(TreeNode*) ;

  void build(const char *filename);

  void BFS();
  void DFS(double *input, std::vector<IntDoubleMap>&, int level,
      std::vector<Condition*>&, FILE *fp);
  void DFS_3(double *input, std::vector<IntDoubleMap>&, int level,
      std::vector<Condition*>&, FILE *fp);
  IntDoubleMap sub_DFS(double *input, TreeNode *root,
      std::vector<Condition*>& problems, FILE *fp,
      std::vector<IntDoubleMap>& highPriorSolutions, int level) ;
  void addNodes();
};

#endif
