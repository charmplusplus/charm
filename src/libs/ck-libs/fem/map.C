
/*
   partdmesh gives info on which PE gets which nodes,
   this program is to find:
   for each PE, does it have a node that needs communication with neighbors?
   If so, what are the node numbers?

   Author:	Rui Liu, Dept Computer Science, Unversity of Illinois, USA
   Started:	July 20, 1999

*/ 

#include "map.h"

static int intcompare(const void *i, const void *j);

main(int argc, char *argv[]) {

  int i, j, k, l, ne, nn, nparts, etype, esize;
  int *elmnts, *epart;
  /*  char etypestr[4][5] = {"TRI", "TET", "HEX", "QUAD"};  */
  FILE *fpmesh, *fpmeshEpart, *fpmeshNpart;
  char filename[256];
  struct MeshInfoPart *meshInfoPart;
  struct ShareNodeList *shareNodeList;
  struct CommInfoPart *commInfoPart;
  struct CommNodeList *commNodeList;
  int **nodeNum;
  int *counterElmnts;
  int counter;
  int *temp2;

  int *xadj, *adjncy;
  int numflag;
  int *node_ptr;

  if (argc != 4) {
    printf("Usage: %s <meshfile> <number of partitions> <element size>\n",
           argv[0]);
    exit(1);
  }

  nparts = atoi(argv[2]);
  esize = atoi(argv[3]);
  	/* 3 for triangle elements, 4 for quadrilaterals and tetrahedra,
           6 for 6-node elements,   8 for hexahedra (bricks). */
  if ( (esize < 3) || (esize > 8) ) {
    printf("Element size probably out of reasonable range.\n");
    printf("This program assumes it to be 3--8.\n");
    exit(1);
  }

  srand48(time(NULL));

  meshInfoPart = new (struct MeshInfoPart)[nparts];
  if (! meshInfoPart) {
    printf("No enough memory for MeshInfoPart.\n");
    exit(1);
  }

  if ((fpmesh = fopen(argv[1], "r")) == NULL) {
    printf("Failed to open file %s\n", argv[1]);
    exit(1);
  }
  fscanf(fpmesh, "%d %d", &ne, &etype);

  sprintf(filename,"%s.epart.%d",argv[1], nparts);
  if ((fpmeshEpart = fopen(filename, "r")) == NULL) {
    printf("Failed to open file %s\n", filename);
    exit(1);
  }

  if (NULL == (counterElmnts = new int[nparts]) ) {
    printf("No enough memory for counterElmnts.\n");
    fclose(fpmeshEpart);
    exit(1);
  }
  epart = new int[ne];
  if (! epart) {
    printf("No enough memory for epart.\n");
    fclose(fpmeshEpart);
    exit(1);
  }
  for (i=0; i<ne; i++)
    fscanf(fpmeshEpart, "%d", epart+i);
  fclose(fpmeshEpart);

  for (i=0; i<nparts; i++)
    meshInfoPart[i].numElements = meshInfoPart[i].numNodes = 0;

  for (i=0; i<ne; i++)
    meshInfoPart[epart[i]].numElements++;
  for (i=0; i<nparts; i++)
    meshInfoPart[i].numNodes = esize * meshInfoPart[i].numElements;

  for (i=0; i<nparts; i++) {
    meshInfoPart[i].nodeNum = new int[meshInfoPart[i].numNodes];
    meshInfoPart[i].elementNum = new int[meshInfoPart[i].numElements];
  }

  if (NULL == (elmnts = new int[esize*ne]) ) {
    printf("No enough memory for elmnts.\n");
    fclose(fpmesh);
    exit(1);
  }
  for (j=esize*ne, i=0; i<j; i++) {
    fscanf(fpmesh, "%d", elmnts+i);
    elmnts[i]--;
  }
  fclose(fpmesh);

  nn = elmnts[0];
  for (i=1; i<j; i++)
    if (nn<elmnts[i])
      nn=elmnts[i];
  nn++;
  /* nn++ to fix the C convention that index starts from 0 */

  for (i=0; i<nparts; i++) counterElmnts[i] = 0;
  for (int temp, temp1, i=0; i<ne; i++) {
    temp = epart[i];   /* temp is the Pe number */
    temp1 = counterElmnts[temp] * esize;
    meshInfoPart[temp].elementNum[counterElmnts[temp]++] = i;
    for (j=0; j<esize; j++)
      meshInfoPart[temp].nodeNum[temp1++] = elmnts[i*esize+j];
  }
/*
  printf("\nElements at Pe:\n");
  for (i=0; i<nparts; i++) {
    printf("Pe[%d]: %d -- ", i, meshInfoPart[i].numElements);
    for (j=0; j<meshInfoPart[i].numElements; j++)
      printf("%d ", meshInfoPart[i].elementNum[j]);
    printf("\n");
  }
*/

  for (i=0; i<nparts; i++)
    qsort(meshInfoPart[i].nodeNum, meshInfoPart[i].numNodes, sizeof(int),
	  intcompare);

  for (int temp, temp1, i=0; i<nparts; i++) {
    temp = 1;	/* temp means number of unique node numbers for a Pe. */
    temp1 = meshInfoPart[i].nodeNum[0];
    for (j=1; j<meshInfoPart[i].numNodes; j++)
      if (meshInfoPart[i].nodeNum[j] != temp1) {
	temp++;
	temp1 = meshInfoPart[i].nodeNum[j];
      }

    temp2 = (int *)malloc(temp*sizeof(int));
    counter = 1;  /* counter now is the position of the new unique node list */
    temp2[0] = temp1 = meshInfoPart[i].nodeNum[0];
    for (j=1; j<meshInfoPart[i].numNodes; j++)
      if (meshInfoPart[i].nodeNum[j] != temp1) {
	temp2[counter++] = meshInfoPart[i].nodeNum[j];
	temp1 = meshInfoPart[i].nodeNum[j];
      }

    meshInfoPart[i].numNodes = temp;
    free(meshInfoPart[i].nodeNum);
    meshInfoPart[i].nodeNum = temp2;

  }

  /* printf("After uniquefying:\n"); */
/*
  printf("\nNodes at Pe:\n");
  for (i=0; i<nparts; i++) {
    printf("Pe[%d]: ", i);
    for (j=0; j<meshInfoPart[i].numNodes; j++)
      printf("%d ", meshInfoPart[i].nodeNum[j]);
    printf("\n");
  }
  printf("\n");
*/

  if (NULL == (xadj = new int[nn+1]) ) {
    printf("No enough memory for xadj.\n");
    exit(1);
  }
  if (NULL == (adjncy = new int[6*nn]) ) {
    printf("No enough memory for adjncy.\n");
    exit(1);
  }

  numflag = 0;	/* c-style array indexing: start from element 0 */

  /* shareNodeList contains information for communication
     it contains the following info:
     for each node, how many Pes share it, and what are the Pe #'s
  */

  // allocation and initialization for shareNodeList
  shareNodeList = new (struct ShareNodeList)[nn];
  if (! shareNodeList) {
    printf("No enough memory for shareNodeList.\n");
    exit(1);
  }
  for (i=0; i<nn; i++)
    shareNodeList[i].number = 0;

  /* scan through the nodes on all Pe's to generate proper shareNodeList */
  for (int temp, temp1, i=0; i<nparts; i++) {
    temp = meshInfoPart[i].numNodes;
    for (j=0; j<temp; j++) {
      temp1 = meshInfoPart[i].nodeNum[j];
      shareNodeList[temp1].Pes[shareNodeList[temp1].number++] = i;
    }
  }

  int *homePe = new int[nn];
  if (!homePe) {
    printf("No enough memory for homePe.\n");
    exit(1);
  }
  for (int i=0; i<nn; i++) {
    int numPes = shareNodeList[i].number; // number of Pes who share node i
    if (1 == numPes) {
      homePe[i] = shareNodeList[i].Pes[0];
    }
    else {
      int randomNum = (int)(drand48()*numPes);
      if (randomNum == numPes)
	randomNum = numPes - 1; // to avoid out of bound for array access
      homePe[i] = shareNodeList[i].Pes[randomNum];
    }
  }

/*
  printf("shareNodeList:\n");
  for (i=0; i<nn; i++) {
    temp = shareNodeList[i].number; 
    printf("%d: %d -- ", i, temp);
    for (j=0; j<temp; j++)
      printf("%d ", shareNodeList[i].Pes[j]);
    printf("\n");
  }
*/
 
  /* commInfoPart contains information for communication
     it contains the following info:
     for each Pe, how many Pes it needs to communicate, 
     the Pe #'s, how many boundary nodes for the corresponding Pe,
     and the local node number of the boundary nodes
  */

  commInfoPart = new (struct CommInfoPart)[nparts];
  if (! commInfoPart) {
    printf("No enough memory for commInfoPart.\n");
    exit(1);
  }
  for (i=0; i<nparts; i++) {
    commInfoPart[i].numPes = 0;
  }

  for (i=0; i<nparts; i++) {
    if (NULL == 
	(commInfoPart[i].commNodeList = new (struct CommNodeList)[nparts]) ) {
      printf("No enough memory for commInfoPart[%d].commNodeList.\n", i);
      exit(1);
    }

    for (j=0; j<nparts; j++) {
      if (NULL == 
	  (commInfoPart[i].commNodeList[j].commNodeNum = new int[meshInfoPart[i].numNodes]) )  {
        printf("No enough memory for commInfoPart[%d].commNodeList[%d].commNodeNum.\n", i, j);
        exit(1);
      }
      commInfoPart[i].commNodeList[j].numCommNodes = 0;
    }

    for (int temp, temp1, k=0; k<meshInfoPart[i].numNodes; k++) {
      temp = meshInfoPart[i].nodeNum[k];
      /* temp is the global node # */
      for (l=0; l<shareNodeList[temp].number; l++) {
	temp1 = shareNodeList[temp].Pes[l];
	if (temp1 != i) {
	  commInfoPart[i].commNodeList[temp1].commNodeNum[commInfoPart[i].commNodeList[temp1].numCommNodes++] = k;
          /* store the local node # k instead of global # */
	}
      }

    }  
    for (j=0; j<nparts; j++) {
      if (commInfoPart[i].commNodeList[j].numCommNodes != 0)
	commInfoPart[i].numPes++;
    }

  }

  for (i=0; i<nparts; i++) {
    sprintf(filename,"%s.out.Pe%d",argv[1], i); /* open file,one for each Pe */
    if ((fpmesh = fopen(filename, "w")) == NULL) {
      printf("Failed to open file %s\n", filename);
      exit(1);
    }
    fprintf(fpmesh, "%d\n", meshInfoPart[i].numNodes);
    for (int temp, j=0; j<meshInfoPart[i].numNodes; j++) {
      temp = meshInfoPart[i].nodeNum[j];
      // fprintf(fpmesh, "%d %d\n", temp+1, (shareNodeList[temp].number == 1));
      fprintf(fpmesh, "%d %d\n", temp+1, homePe[temp]);
    }
    /* local-global node number mapping, and exclusive/share flag */
    fprintf(fpmesh, "%d %d\n", meshInfoPart[i].numElements, esize);
    for (int temp, j=0; j<meshInfoPart[i].numElements; j++) {
      fprintf(fpmesh, "%d ", meshInfoPart[i].elementNum[j]+1);
      temp = esize * meshInfoPart[i].elementNum[j];

      for (k=0; k<esize; k++) {
	temp2 = 
	  bsearch(&elmnts[temp+k], meshInfoPart[i].nodeNum,
		  meshInfoPart[i].numNodes, sizeof(int), intcompare);
	if (NULL != temp2)
	  fprintf(fpmesh, "%d ", (temp2 - meshInfoPart[i].nodeNum));
	else {
	  printf("bsearch failed to find the global node number.\n");
	  exit(1);
	}
      }
      fprintf(fpmesh, "\n");
    }
    fprintf(fpmesh, "%d\n", commInfoPart[i].numPes);
    for (int temp, j=0; j<nparts; j++) {
      temp = commInfoPart[i].commNodeList[j].numCommNodes;
      if (temp != 0) {
	fprintf(fpmesh, "%d %d ", j, temp);
      for (k=0; k<temp; k++)
	fprintf(fpmesh, "%d ", commInfoPart[i].commNodeList[j].commNodeNum[k]);
      fprintf(fpmesh, "\n");
      }
    }

    fclose(fpmesh);
  }

  for (i=0; i<nparts; i++) {
    for (j=0; j<nparts; j++)
      delete commInfoPart[i].commNodeList[j].commNodeNum;
    delete commInfoPart[i].commNodeList;
  }
  delete commInfoPart;

  delete shareNodeList;

  delete adjncy;
  delete xadj;
  delete epart;
  delete counterElmnts;
  for (i=0; i<nparts; i++) {
    delete meshInfoPart[i].nodeNum;
    delete meshInfoPart[i].elementNum;
  }
  delete elmnts;
  delete meshInfoPart;

  delete homePe;
}

static int intcompare(const void *i, const void *j) {
  if ((*(int *)i) > (*(int *)j))
    return (1);
  else if ((*(int *)i) < (*(int *)j))
    return (-1);
  return (0);
}
