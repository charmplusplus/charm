
#include <stdio.h>
#include <stdlib.h>
/* the following two are used for time(NULL) to get time */
#include <sys/types.h>
#include <time.h>

struct Node {
  int globleNum;
  int exclusive;
  int numNeighbors;
  int *neighborList;
};

struct MeshInfoPart {
  int numElements;
  int numNodes;
  int *elementNum;
  int *nodeNum;
  struct Node *nodeList;
};

struct ShareNodeList {
  int number;
  int Pes[10];
};
/* shareNodeList contains information for communication
   it contains the following info:
   for each node, how many Pes share it, and what are the Pe #'s
*/

struct CommNodeList {
  int numCommNodes;
  int *commNodeNum;
};

struct CommInfoPart {
  int numPes;
  struct CommNodeList *commNodeList;
};
/* commInfoPart contains information for communication
   it contains the following info:
   for each Pe, how many Pes it needs to communicate, 
   the Pe #'s, how many boundary nodes for the corresponding Pe,
   and the local node number of the boundary nodes
*/
