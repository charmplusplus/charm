#ifndef __DECISIONTREE__H__
#define __DECISIONTREE__H__

#include <vector>
#include <map>
#include "picstreenode.h"
#include "charm++.h"

using namespace std;

class DecisionTree {

  TreeNode *root;

public:

  DecisionTree() ;
  DecisionTree(TreeNode*) ;

  void build(char *filename);

  void BFS();
  void DFS(double *input, vector<IntDoubleMap>&, int level, std::vector<Condition*>&, FILE *fp);
  void DFS_3(double *input, vector<IntDoubleMap>&, int level, std::vector<Condition*>&, FILE *fp);
  IntDoubleMap sub_DFS(double *input, TreeNode *root, std::vector<Condition*>& problems, FILE *fp, vector<IntDoubleMap>& highPriorSolutions, int level) ;
  void addNodes();
};

#endif
