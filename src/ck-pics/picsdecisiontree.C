#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <stack>
#include <map>
#include "picsdefs.h"
#include "picsdecisiontree.h"

#define NAME_LENGTH 30

DecisionTree::DecisionTree() {
  root=NULL;
}

DecisionTree::DecisionTree(TreeNode *p) {
  root = p;
}

typedef std::map<std::string, int> keyid_map;
typedef std::map<std::string, TreeNode*> tree_map;

void DecisionTree::build(const char *filename) {

  keyid_map fieldMap;
  keyid_map updownMap;
  keyid_map effectMap;
  keyid_map symbolMap;
  keyid_map opMap;
  //setup map
  for(int i=0; i<NUM_NODES; i++) { fieldMap[FieldName[i]] = i; }
  for(int i=0; i<PICS_NUM_EFFECTS; i++) { effectMap[EffectName[i]] = i; }

  updownMap["UP"] = UP;
  updownMap["DOWN"] = DOWN;

  symbolMap["IS"] = 0;
  symbolMap["LT"] = 1;
  symbolMap["GT"] = 2;
  symbolMap["NLT"] = 3;
  symbolMap["NGT"] = 4;
  symbolMap["NOTIS"] = 5;

  opMap["ADD"] = 0;
  opMap["SUB"] = 1;
  opMap["MUL"] = 2;
  opMap["DIV"] = 3;

  std::ifstream file(filename);

  int nodeType;
  std::string keyStr;
  std::string fieldTypeName;
  std::string symbol;
  std::string parentName;
  std::string typeStr;
  std::string baseFieldType;
  std::string op;
  int  fieldType;
  int  flag;
  std::string avgMinMax;
  size_t len = 0;
  std::string line;
  tree_map nodemap;
  TreeNode *node;
  int numOfParents;
  double threshold;
  float base;
  Condition *cond;
  Solution *sol;
  int sumbytes=0;
  int bytes;

  while (std::getline(file, line)) {
    if(line[0] == '#')
      continue;
    std::istringstream stream(line);
    stream >> nodeType >> keyStr >> typeStr >> fieldTypeName >> bytes;

    switch(nodeType) {
    case -1:        //root
      root = nodemap["Root"]= new TreeNode(NULL, new Condition());
      break;

    case 0:     //internal node
      fieldType = fieldMap[typeStr + "_" + fieldTypeName];
      stream >> op >> bytes;
      stream >> flag >> bytes;
      if(flag == -1)
      {
        stream >> base >> symbol >> numOfParents >> parentName >> bytes;
        cond = new Condition(keyStr, fieldType, (Operator)opMap[op], base, (CompareSymbol)(symbolMap[symbol]));
        nodemap[keyStr] = new TreeNode(nodemap[parentName], cond);
        nodemap[parentName]->addChild(nodemap[keyStr]);
      }else if(flag == 0)
      {
        stream >> avgMinMax >> baseFieldType >> symbol >> threshold >> numOfParents >> parentName >> bytes;
        cond = new Condition(keyStr, fieldType, (Operator)opMap[op], fieldMap[avgMinMax + "_" + baseFieldType], threshold,  (CompareSymbol)symbolMap[symbol]);
        node =  new TreeNode(nodemap[parentName], cond);
        nodemap[keyStr] = node;
        nodemap[parentName]->addChild(nodemap[keyStr]);
      }
      break;

    case 1:     //leaf
      stream >> numOfParents >> parentName >> bytes;
      sol = new Solution( (Direction)updownMap[typeStr], (Effect)effectMap[fieldTypeName]);
      node = new TreeNode(nodemap[parentName], sol);
      nodemap[parentName]->addChild(node);
      for(int i=1; i<numOfParents; i++) {
        stream >> parentName >> bytes;
        node = new TreeNode(nodemap[parentName], sol);
        nodemap[parentName]->addChild(node);
      }
      break;
    }
  };
}

void DecisionTree::addNodes() {
}

void DecisionTree::BFS() {
  TreeNode *current = NULL;
  TreeNode *child = NULL;
  std::queue<TreeNode*> myqueue;
  myqueue.push(root);
  while(!myqueue.empty())
  {
    current = myqueue.front();
    myqueue.pop();
    printf("{");
    if(current->getParent()!=NULL)
      current->getParent()->printMe();
    current->printMe();
    printf("}\n");
    for(current->beginChild(); !(current->isEndChild()); current->nextChild())
    {
      child = current->getCurrentChild();
      myqueue.push(child);
    }
  };
}

void DecisionTree::DFS( double *input, std::vector<IntDoubleMap>& solutions,
    int level, std::vector<Condition*>& problems, FILE *fp) {
  std::stack<TreeNode*> mystack;
  TreeNode *current = NULL;
  TreeNode *child = NULL;
  mystack.push(root);
  while(!mystack.empty()) {
    current = mystack.top();
    mystack.pop();
    for(current->beginChild(); !(current->isEndChild()); current->nextChild())
    {
      child = current->getCurrentChild();
      if(child->isSolution())
      {
        int effect = child->getSolutionValue();
        int ignore = 0;
        //check higher priority solution for conflicts
        for(int higher=0; higher<level; higher++){
            if(solutions[higher].count(-effect) > 0 || solutions[higher].count(effect) > 0)
            {
                ignore = 1;
                break;
            }
        }
        if(!ignore) {
            if(solutions[level].count(-effect) > 0){
                //reverse effect exist, keep the one with larger performance improvement
                if(current->getPotentialImprove() > solutions[level][-effect])
                {
                    printf("\n-----detected conflict effects------ reverse %d %f \n", -effect, solutions[level][-effect]);
                    solutions[level].erase(-effect);
                    solutions[level][effect] = current->getPotentialImprove();
                    child->printDataToFile(input, fp);
                }
            }
            else
            {
                solutions[level][child->getSolutionValue()] = current->getPotentialImprove();
                child->printDataToFile(input, fp);
            }
        }
      }
      else
      {
        if(child->test(input))
        {
          mystack.push(child);
          problems.push_back(child->getValue().condition);
          if(child->getPotentialImprove()==-100)
            child->setPotentialImprove(current->getPotentialImprove());
          child->printDataToFile(input, fp);
        }
      }
    }
  };
}

//keep the ones without conflict, keep the conflicted one for the problem without other solutions
//(A,B,C) (B) --> (A,C) (B)
void DecisionTree::DFS_3( double *input, std::vector<IntDoubleMap>& solutions, int level, std::vector<Condition*>& problems, FILE *fp) {
  TreeNode *child = NULL;
  std::vector<IntDoubleMap> rawSolutions;
  for(root->beginChild(); !(root->isEndChild()); root->nextChild())
  {
    child = root->getCurrentChild();
    if(child->test(input))
    {
      //perform here
      child->printDataToFile(input, fp);
      rawSolutions.push_back(sub_DFS(input, child, problems, fp, solutions, level));
    }
  }

  //keep the ones without conflict
  std::vector<int> hasSolutions(rawSolutions.size(), 0);
  IntDoubleMap &results = solutions[level];
  for(int i=0; i<rawSolutions.size();i++){
    //if there is only one solution, keep it
    if(rawSolutions[i].size() == 1)
    {
      IntDoubleMap::iterator iter=rawSolutions[i].begin();
      results[iter->first] = iter->second;
      hasSolutions[i] = 1;
      continue;
    }
    //keep the solutions without conflict
    for(IntDoubleMap::iterator iter=rawSolutions[i].begin(); iter!= rawSolutions[i].end(); iter++)
    {
      int eff = iter->first;
      bool hasConflict = false;
      for(int j=0; j<rawSolutions.size();j++){
        if(rawSolutions[j].count(-eff))
        {
          hasConflict = true;
          break;
        }
      }
      if(!hasConflict)
      {
        results[eff] = iter->second;
        hasSolutions[i] = 1;
      }
      else{
        //check whether it exists in the high level solution set, if it does, keep it
        for(int i=0; i<level; i++){
        if(solutions[i].count(eff)>0){
          results[eff] = iter->second;
          hasSolutions[i] = 1;
          break;
        }
        }
      }
    }
  }

  //try to assign at least one solution to a problem category
  for(int i=0; i<rawSolutions.size();i++){
    if(hasSolutions[i])
      continue;
    for(IntDoubleMap::iterator iter=rawSolutions[i].begin(); iter!= rawSolutions[i].end(); iter++)
    {
      int eff = iter->first;
      bool hasConflict = false;
      if(results.count(eff)>0)
      {
        hasSolutions[i]= 1;
        break;
      }
      else if(results.count(-eff)==0){
          results[eff] = iter->second;
          hasSolutions[i]= 1;
          break;
      }
    }
  }

  //phase 3, for the other solutions, so long as there is no conflict, keep them
  for(int i=0; i<rawSolutions.size();i++){
    for(IntDoubleMap::iterator iter=rawSolutions[i].begin(); iter!= rawSolutions[i].end(); iter++)
    {
      int eff = iter->first;
      bool hasConflict = false;
      if(results.count(-eff)==0){
        results[eff] = iter->second; 
      }
    }
  }

  //result has the solutions
}

IntDoubleMap DecisionTree::sub_DFS(double *input, TreeNode *root,
    std::vector<Condition*>& problems, FILE *fp,
    std::vector<IntDoubleMap>& highPriorSolutions, int level) {
  TreeNode *current = NULL;
  TreeNode *child = NULL;
  std::stack<TreeNode*> mystack;
  IntDoubleMap solutions;

  mystack.push(root);
  while(!mystack.empty()) {
    current = mystack.top();
    mystack.pop();
    for(current->beginChild(); !(current->isEndChild()); current->nextChild())
    {
      child = current->getCurrentChild();
      if(child->isSolution())
      {
        int effect = child->getSolutionValue();
        int ignore = 0;
        //check higher priority solution for conflicts
        for(int higher=0; higher<level; higher++){
            if(highPriorSolutions[higher].count(-effect) > 0 )
            {
                ignore = 1;
                break;
            }
        }
        if(!ignore){
          solutions[effect] = current->getPotentialImprove();
          child->printDataToFile(input, fp);
        }
      }
      else
      {
        if(child->test(input))
        {
          mystack.push(child);
          problems.push_back(child->getValue().condition);
          if(child->getPotentialImprove()==-100)
            child->setPotentialImprove(current->getPotentialImprove());
          child->printDataToFile(input, fp);
        }
      }
    }
  };
  return solutions;
}


