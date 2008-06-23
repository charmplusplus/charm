#include <ParFUM_TOPS.h>
#include <ParFUM.h>
//#include "dataType.h"
#include <iostream>


char* default_filename = "input.txt";


// 4 node tets
#define nelnode  4

typedef struct
{
} MatProperty;

typedef struct
{
} ElemAtt;

typedef struct
{
} NodeAtt;

typedef struct
{
} ModelAtt;


#undef DEBUG
#define DEBUG 0

void addElement(TopModel *model, int id, int n0, int n1, int n2, int n3){

#if DEBUG
	CkPrintf("ele[%d] is %d %d %d %d\n", id, n0, n1, n2, n3);
#endif

	TopNode ninc[4];
	TopElement elem;
	ElemAtt*  eAtt;

	ninc[0] = topModel_GetNodeAtId (model,n0);
	ninc[1] = topModel_GetNodeAtId (model,n1);
	ninc[2] = topModel_GetNodeAtId (model,n2);
	ninc[3] = topModel_GetNodeAtId (model,n3);

	// Adds element to the model
	elem = topModel_InsertElem (model,TOP_ELEMENT_TET4,ninc);
	topElement_SetId (model,elem,id);

}

void readMeshFromFile(TopModel* model, char* filename){
	int nn, ne, i, id;
	char commend[512];

	FILE* fp;
	fp = fopen (filename, "r");
	if (fp==NULL) {
		fprintf(stderr,"Could not open file: %s.\n",filename);
		exit(1);
	}

	// Reads nodes (id, x, y, z)
	fscanf(fp, "%s \n", commend);
	if (!strcmp (commend, "Node\n")) {
		printf("Error in reading Node \n");
		exit(1);
	}

	if (fscanf(fp,"%d", &nn) != 1) {
		printf("Invalid format of node file.\n");
		exit(1);
	}

	for (i = 0; i < nn; i++)
	{
		double x, y, z;
		int id;
		TopNode node;
		NodeAtt*  nAtt;
		if (fscanf(fp,"%d, %lf, %lf, %lf",&id, &x, &y, &z) != 4) {
			fprintf(stderr,"Invalid format for nodes.\n");
			exit(1);
		}
		z = 0;
		// Adds node to the model
		node = topModel_InsertNode (model, x, y, z);
		topNode_SetId (model, node, id);
		nAtt = (NodeAtt*) malloc(sizeof(NodeAtt));
		assert(nAtt);

	}
	printf ("The number of nodes: %d \n", nn);


	fscanf(fp, "%s \n", commend);
	if (!strcmp (commend, "Element\n"))
	{
		printf("Error in reading Element \n");
		exit(1);
	}

	if (fscanf(fp,"%d", &ne) != 1) {
		printf("Invalid format of element file.\n");
		exit(1);
	}

	for (i = 0; i < ne; i++)
	{
		int matnum = 1;
		int j, id;
		int inc[4];
		TopNode ninc[4];
		TopElement elem;
		ElemAtt*  eAtt;

		if (fscanf(fp,"%d, %d, %d, %d, %d",&id,&inc[0],&inc[1],&inc[2],&inc[3]) != 5)
		{
			fprintf(stderr,"Invalid format for element.\n");
			exit(1);
		}

		for (j=0; j<nelnode; ++j)
			ninc[j] = topModel_GetNodeAtId (model,inc[j]);

		// Adds element to the model
		elem = topModel_InsertElem (model,TOP_ELEMENT_TET4,ninc);
		topElement_SetId (model,elem,id);

		// Initialize element attribute ( CANNOT BE DONE IN INIT )s
		//		eAtt = (ElemAtt*) malloc(sizeof(ElemAtt));
		//		assert(eAtt);
		//		initElemAtt(eAtt);
		//		topElement_SetAttrib (model, elem, eAtt);
	}
	printf ("The number of elements: %d \n", ne);


}




/// Generate a torus mesh where each brick is subdivided into 6 tets
void generateTorusMesh6(TopModel* model){


	// Generate a simple torus mesh
	CkPrintf("Generating a simple torus mesh...\n");
	int sx, sy, sz; // unit size of cube mesh
	sx = 60; sy = 11; sz = 11;
	int nPts = sx*sy*sz;
	int nEle = ((sx)*(sy-1)*(sz-1))*5;

	/// The inner radius
	double r1 = 500.0;
	/// The outer radius (actually we should adjust sy below to make this the actual outer radius)
	double r2 = 1500.0;
	
	double thickness = 1000.0;

	for (int id = 0; id < nPts; id++)
	{
		double x, y, z;
		TopNode node;
		NodeAtt*  nAtt;
		// Adds node to the model
		node = topModel_InsertNode (model, x, y, z);
		topNode_SetId (model, node, id);
	}	  

	cout << "Added " << nPts << " nodes" << endl;

	// Set node coordinates
	for (int x=0; x<sx; x++) {
		for (int y=0; y<sy; y++) {
			for (int z=0; z<sz; z++) {

				double pi = 3.1416;
				double theta =  ((double)x) / ((double)sx) * 2.0 * pi;
				double r = r1 + ((double)y/(double)sy)*(r2-r1);

				double x_coord = r * sin(theta);
				double y_coord = r * cos(theta); 
				double z_coord = thickness * (double)z / (double)sz;

				int nCount =  x*sy*sz + y*sz + z ;
				TopNode node = topModel_GetNodeAtId(model,nCount);

				(*model->coord_T)(node,0) = x_coord;
				(*model->coord_T)(node,1) = y_coord;
				(*model->coord_T)(node,2) = z_coord;

			}
		}
	}


	// Create elements
	int eCount=0; 
	for (int x=0; x<sx; x++) { 
		for (int y=0; y<sy-1; y++) { 
			for (int z=0; z<sz-1; z++) { 

				/*	      

					     3 4
				         1 2

				         7 8
					     5 6

					     tets: 1568 1758   1628 1248   1438 1378 
				 */

				// nCount = x*sy*sz + y*sz + z

				int n1 = x*sy*sz     + y*sz     + z     ;
				int n2 = x*sy*sz     + y*sz     + (z+1) ;
				int n3 = x*sy*sz     + (y+1)*sz + z     ;
				int n4 = x*sy*sz     + (y+1)*sz + (z+1) ;
				int n5 = (x+1)*sy*sz + y*sz     + z     ;
				int n6 = (x+1)*sy*sz + y*sz     + (z+1) ;
				int n7 = (x+1)*sy*sz + (y+1)*sz + z     ;
				int n8 = (x+1)*sy*sz + (y+1)*sz + (z+1) ;

				// If this is the max x plane, wrap around
				if(x==sx-1){
					n5 = 0*sy*sz     + y*sz     + z     ;
					n6 = 0*sy*sz     + y*sz     + (z+1) ;
					n7 = 0*sy*sz     + (y+1)*sz + z     ;
					n8 = 0*sy*sz     + (y+1)*sz + (z+1) ;

					CkAssert(n5 < sy*sz);
					CkAssert(n6 < sy*sz);
					CkAssert(n7 < sy*sz); 
					CkAssert(n8 < sy*sz); 

				}


				// 1568 1758  
				addElement(model, eCount++, n1, n5, n6, n8);
				addElement(model, eCount++, n1, n7, n5, n8);

				// 1628 1248   
				addElement(model, eCount++, n1, n6, n2, n8);
				addElement(model, eCount++, n1, n2, n4, n8);

				// 1438 1378 
				addElement(model, eCount++, n1, n4, n3, n8);
				addElement(model, eCount++, n1, n3, n7, n8);

			}

		}
	}
}




/// Generate a mesh that is a topologically a cube with one face incident to its opposite face. Wrap around in x dimension
void generateTorusMesh(TopModel* model){


	// Generate a simple torus mesh
	CkPrintf("Generating a simple torus mesh...\n");
	int sx, sy, sz; // unit size of cube mesh
	sx = 60; sy = 11; sz = 11;
	int nPts = sx*sy*sz;
	int nEle = ((sx)*(sy-1)*(sz-1))*5;

	CkAssert(sx % 2 == 0);
	

	/// The inner radius
	double r1 = 500.0;
	/// The outer radius (actually we should adjust sy below to make this the actual outer radius)
	double r2 = 700.0;


	for (int id = 0; id < nPts; id++)
	{
		double x, y, z;
		TopNode node;
		NodeAtt*  nAtt;
		// Adds node to the model
		node = topModel_InsertNode (model, x, y, z);
		topNode_SetId (model, node, id);
	}	  

	cout << "Added " << nPts << " nodes" << endl;

	// Set node coordinates
	for (int x=0; x<sx; x++) {
		for (int y=0; y<sy; y++) {
			for (int z=0; z<sz; z++) {

				double pi = 3.1416;
				double theta =  ((double)x) / ((double)sx) * 2.0 * pi;
				double r = r1 + ((double)y/(double)sy)*(r2-r1);

				double x_coord = r * sin(theta);
				double y_coord = r * cos(theta); 
				double z_coord = 100.0*z;

				int nCount =  x*sy*sz + y*sz + z ;
				TopNode node = topModel_GetNodeAtId(model,nCount);

				(*model->coord_T)(node,0) = x_coord;
				(*model->coord_T)(node,1) = y_coord;
				(*model->coord_T)(node,2) = z_coord;

			}
		}
	}


	// Create elements
	int eCount=0; 
	for (int x=0; x<sx; x++) { 
		for (int y=0; y<sy-1; y++) { 
			for (int z=0; z<sz-1; z++) { 

				/*	     y   
						    /    
					       /     
					      /	     
					     /______z
					     | 	    
					     |	  
					     | 	  
					     |	  
					     x 	  



					     TOP:
					     nCount+sz         nCount+sz+1
					     nCount            nCount+1

					     BOTTOM:
					     nCount+sy*sz+sz   nCount+sy*sz+sz+1
					     nCount+sy*sz      nCount+sy*sz+1


					     3 4
				         1 2

				         7 8
					     5 6

					     tets: 1235 2438 5628 5873 2358
				 */

				// nCount = x*sy*sz + y*sz + z

				int n1 = x*sy*sz     + y*sz     + z     ;
				int n2 = x*sy*sz     + y*sz     + (z+1) ;
				int n3 = x*sy*sz     + (y+1)*sz + z     ;
				int n4 = x*sy*sz     + (y+1)*sz + (z+1) ;
				int n5 = (x+1)*sy*sz + y*sz     + z     ;
				int n6 = (x+1)*sy*sz + y*sz     + (z+1) ;
				int n7 = (x+1)*sy*sz + (y+1)*sz + z     ;
				int n8 = (x+1)*sy*sz + (y+1)*sz + (z+1) ;

				// If this is the max x plane, wrap around
				if(x==sx-1){
					n5 = 0*sy*sz     + y*sz     + z     ;
					n6 = 0*sy*sz     + y*sz     + (z+1) ;
					n7 = 0*sy*sz     + (y+1)*sz + z     ;
					n8 = 0*sy*sz     + (y+1)*sz + (z+1) ;

					CkAssert(n5 < sy*sz);
					CkAssert(n6 < sy*sz);
					CkAssert(n7 < sy*sz); 
					CkAssert(n8 < sy*sz); 

				}


				bool even = (((int)(x+y+z)%2) == 0); 
				if (even) {

					// 1235
					addElement(model, eCount++, n1, n2, n3, n5);

					// 2438
					addElement(model, eCount++, n2, n4, n3, n8);

					// 5628
					addElement(model, eCount++, n5, n6, n2, n8);

					// 5837
					addElement(model, eCount++, n5, n8, n3, n7);

					// 2358
					addElement(model, eCount++, n2, n3, n5, n8);


				}
				else {
					// 1246						 
					addElement(model, eCount++, n1, n2, n4, n6);

					// 1567
					addElement(model, eCount++, n1, n5, n6, n7);

					// 1743
					addElement(model, eCount++, n1, n7, n4, n3);

					// 4768
					addElement(model, eCount++, n4, n7, n6, n8);

					// 1476
					addElement(model, eCount++, n1, n4, n7, n6);

				}

			}
		}

	}

}





/// Generate a mesh that is a topologically a cube with one face incident to its opposite face.
void generateCubeMesh(TopModel* model){

	int nPts = 0; //Number of nodes
	int nEle = 0; // Number of tets


	// Generate a simple cube mesh
	CkPrintf("Generating a simple cube mesh...\n");
	int sx, sy, sz; // unit size of cube mesh
	sx = 8; sy = 8; sz = 8;
	nPts = sx*sy*sz;
	nEle = ((sx-1)*(sy-1)*(sz-1))*5;


	for (int id = 0; id < nPts; id++)
	{
		double x, y, z;
		TopNode node;
		NodeAtt*  nAtt;
		// Adds node to the model
		node = topModel_InsertNode (model, x, y, z);
		topNode_SetId (model, node, id);
	}	  

	int nCount=0;
	int eCount=0;
	for (double x=0.0; x<(double)sx; x+=1.0) {
		for (double y=0.0; y<(double)sy; y+=1.0) {
			for (double z=0.0; z<(double)sz; z+=1.0) {

				bool even = (((int)(x+y+z)%2) == 0);


				TopNode node = topModel_GetNodeAtId(model,nCount);
				(*model->coord_T)(node,0) = x;
				(*model->coord_T)(node,1) = y;
				(*model->coord_T)(node,2) = z;


/*	     y   
						    /    
					       /     
					      /	     
					     /______z
					     | 	    
					     |	  
					     | 	  
					     |	  
					     x 	  



					     TOP:
					     nCount+sz         nCount+sz+1
					     nCount            nCount+1

					     BOTTOM:
					     nCount+sy*sz+sz   nCount+sy*sz+sz+1
					     nCount+sy*sz      nCount+sy*sz+1


					     3 4
				         1 2

				         7 8
					     5 6

					     tets: 1235 2438 5628 5873 2358
 */


int n1 = nCount;
int n2 = nCount+1;
int n3 = nCount+sz;
int n4 = nCount+sz+1;
int n5 = nCount+sy*sz;
int n6 = nCount+sy*sz+1;
int n7 = nCount+sy*sz+sz;
int n8 = nCount+sy*sz+sz+1;

if (even) {
	if ((x<sx-1.0) && (y<sy-1.0) && (z<sz-1.0)) {

		// 1235
		addElement(model, eCount++, n1, n2, n3, n5);

		// 2438
		addElement(model, eCount++, n2, n4, n3, n8);

		// 5628
		addElement(model, eCount++, n5, n6, n2, n8);

		// 5837
		addElement(model, eCount++, n5, n8, n3, n7);

		// 2358
		addElement(model, eCount++, n2, n3, n5, n8);

	}
}
else {
	if ((x<sx-1.0) && (y<sy-1.0) && (z<sz-1.0)) {
		// 1246						 
		addElement(model, eCount++, n1, n2, n4, n6);

		// 1567
		addElement(model, eCount++, n1, n5, n6, n7);

		// 1743
		addElement(model, eCount++, n1, n7, n4, n3);

		// 4768
		addElement(model, eCount++, n4, n7, n6, n8);

		// 1476
		addElement(model, eCount++, n1, n4, n7, n6);

	}
}
nCount++;  
			}
		}

	}

}












void printNodes(TopModel *model){

	TopNodeItr* n_itr = topModel_CreateNodeItr(model);
	for(topNodeItr_Begin(n_itr);topNodeItr_IsValid(n_itr);topNodeItr_Next(n_itr)){
		TopNode node = topNodeItr_GetCurr(n_itr);
		int x = (*model->coord_T)(node,0);
		int y = (*model->coord_T)(node,1);
		int z = (*model->coord_T)(node,2);

		cout << " === at " << x << "\t" << y << "\t" << z << " is node " << node << endl; 

	}

}


#include "netfem.h"

void myOutputNetFEM(TopModel* model){

	CkPrintf("Writing output via NetFEM\n");

	int time=0;
	int dim = 3;

	NetFEM n=NetFEM_Begin(FEM_My_partition(),time,dim,NetFEM_WRITE);


	int nnodes = topModel_GetNNodes(model);
	double *nodeCoords = new double[3*nnodes];

	int idx = 0;
	TopNodeItr* n_itr = topModel_CreateNodeItr(model);
	for(topNodeItr_Begin(n_itr);topNodeItr_IsValid(n_itr);topNodeItr_Next(n_itr)){
		TopNode node = topNodeItr_GetCurr(n_itr);

		int x = (*model->coord_T)(node,0);
		int y = (*model->coord_T)(node,1);
		int z = (*model->coord_T)(node,2);

		nodeCoords[3*idx + 0] = x;
		nodeCoords[3*idx + 1] = y;
		nodeCoords[3*idx + 2] = z;

		idx++;
	}
	CkAssert(idx == nnodes);

	int nelems = model->mesh->elem[TOP_ELEMENT_TET4].size();

	AllocTable2d<int> *table =  &((FEM_IndexAttribute*)model->mesh->elem[TOP_ELEMENT_TET4].lookup(FEM_CONN,""))->get();
	const int * conn = table->getData();

	int * newconn = new int[nelems*4];
	for(int i=0;i<nelems;i++){
		newconn[4*i+0] = conn[4*i+0];
		newconn[4*i+1] = conn[4*i+1]; 
		newconn[4*i+2] = conn[4*i+3]; 
		newconn[4*i+3] = conn[4*i+2]; 
	}



	NetFEM_Nodes(n,nnodes,nodeCoords,"Position (m)");
	//	  NetFEM_Vector(n,(double *)g.d,"Displacement (m)");
	//	  NetFEM_Vector(n,(double *)g.v,"Velocity (m/s)");

	NetFEM_Elements(n,nelems,4,newconn,"TOP_ELEMENT_TET4");
	//	  NetFEM_Scalar(n,g.S11,1,"X Stress (pure)");
	//	  NetFEM_Scalar(n,g.S22,1,"Y Stress (pure)");
	//	  NetFEM_Scalar(n,g.S12,1,"Shear Stress (pure)");

	NetFEM_End(n);


}




void ensureNoDuplicateNodes(TopModel *model){

	int nnodes = topModel_GetNNodes(model);
	int nodeoccurrences[nnodes];

	for(int i=0;i<nnodes;i++){
		nodeoccurrences[i] = 0;
	}

	TopElemItr* e_itr = topModel_CreateElemItr(model);
	for(topElemItr_Begin(e_itr);topElemItr_IsValid(e_itr);topElemItr_Next(e_itr)){
		TopElement elem = topElemItr_GetCurr(e_itr);

		int etype = elem.type;
		int e = elem.id;

		if(etype == TOP_ELEMENT_TET4 ){
			int n1 = topElement_GetNode(model, elem, 0);
			int n2 = topElement_GetNode(model, elem, 1);
			int n3 = topElement_GetNode(model, elem, 2);
			int n4 = topElement_GetNode(model, elem, 3);	    	  
#if DEBUG			
			CkPrintf("TOP_ELEMENT_TET4 Element %d is adjacent to nodes: %d, %d, %d, %d\n", e, n1, n2, n3, n4);
#endif
			nodeoccurrences[n1] ++;
			nodeoccurrences[n2] ++;
			nodeoccurrences[n3] ++;
			nodeoccurrences[n4] ++;

		}
	}

	bool errordetected = false;

	for(int i=0;i<nnodes;i++){
		if(nodeoccurrences[i] != 1){
			CkPrintf("ERROR: node %d occurs %d times(should only occur once)\n", i, nodeoccurrences[i]);
			errordetected = true;
		}
	}

	if(errordetected)
		CkAbort("Some node didn't get split enough. See above notes");


}




void printElems(TopModel *model){

	TopElemItr* e_itr = topModel_CreateElemItr(model);
	for(topElemItr_Begin(e_itr);topElemItr_IsValid(e_itr);topElemItr_Next(e_itr)){
		TopElement elem = topElemItr_GetCurr(e_itr);

		int etype = elem.type;
		int e = elem.id;

		if(etype == TOP_ELEMENT_TET4 ){
			int n1 = topElement_GetNode(model, elem, 0);
			int n2 = topElement_GetNode(model, elem, 1);
			int n3 = topElement_GetNode(model, elem, 2);
			int n4 = topElement_GetNode(model, elem, 3);	    	  
			CkPrintf("TOP_ELEMENT_TET4 Element %d is adjacent to nodes: %d, %d, %d, %d\n", e, n1, n2, n3, n4);
		} else if (etype == TOP_ELEMENT_COH3T3){
			int n1 = topElement_GetNode(model, elem, 0);
			int n2 = topElement_GetNode(model, elem, 1);
			int n3 = topElement_GetNode(model, elem, 2);
			int n4 = topElement_GetNode(model, elem, 3);
			int n5 = topElement_GetNode(model, elem, 4);
			int n6 = topElement_GetNode(model, elem, 5);
			CkPrintf("TOP_ELEMENT_COH3T3 Element %d is adjacent to nodes: %d, %d, %d, %d, %d, %d\n", e, n1, n2, n3, n4, n5, n6);
		}


	}


}

void init(void)
{
#if DEBUG
	CkPrintf("Size Of: ModelAtt=%d, NodeAtt=%d, ElemAtt=%d\n", 
			sizeof(ModelAtt), sizeof(NodeAtt), sizeof(ElemAtt));
	CkPrintf("Sizeof(FP_TYPE_LOW)=%d, Sizeof(FP_TYPE_HIGH)=%d, \n", 
			sizeof(FP_TYPE_LOW), sizeof(FP_TYPE_HIGH));
#endif

	// Create model & initialize
	TopModel* model =  topModel_Create_Init(4);


//	generateTorusMesh(model);
	generateTorusMesh6(model);
	//	readMeshFromFile(model, "input.txt");

	int nelem = topModel_GetNElem(model);
	int nnodes = topModel_GetNNodes(model);

	printf ("The number of elements: %d \n", nelem);
	printf ("The number of nodes: %d \n", nnodes);

}


void driver(void){


	int myId = FEM_My_partition();
	ModelAtt mAtt;

	double starttime = FEM_Timer();
	TopModel* model =  topModel_Create_Driver(sizeof(ElemAtt),sizeof(NodeAtt), sizeof(ModelAtt), &mAtt);
	double endtime = FEM_Timer();

	cout << "topModel_Create_Driver took: " << endtime-starttime << " seconds" << endl;
	

#if DEBUG
	printElems(model);
#endif



#if 0
	TopElemItr* e_itr = topModel_CreateElemItr(model);
	for(topElemItr_Begin(e_itr);topElemItr_IsValid(e_itr);topElemItr_Next(e_itr)){
		TopElement elem = topElemItr_GetCurr(e_itr);

		int etype = elem.type;
		int e = elem.id;

		CkPrintf("Element %d, %d is adjacent to:\n", etype,e);
		if(etype == TOP_ELEMENT_TET4 ){
			for(int i=0;i<4;i++){
				TopElement neighbor = model->mesh->e2e_getElem(e, i, etype);
				if(neighbor.id != -1){
					CkPrintf("\t");
					neighbor.println(); 
				}
			}
		}  
	}
	CkPrintf("Completed Element adjacency Test\n");
#endif


#if 0
	
	topModel_TestIterators(model);
	
	{
		TopFacetItr*  itr = topModel_CreateFacetItr (model);
		topFacetItr_Begin(itr);
		int facet_count = 0;

		while(topFacetItr_IsValid(itr)){
			TopFacet facet = topFacetItr_GetCurr (itr);
			facet_count = facet_count+1;
			topFacetItr_Next(itr);
		}
		topFacetItr_Destroy (itr);
		cout << "Found " << facet_count << " facets" << endl;
		cout << "Completed Facet iterator test" << endl;
	}
#endif

#if 0
	{
		int countvalid = 0;
		int countall = 0;

		TopFacetItr *itr = topModel_CreateFacetItr (model);
		topFacetItr_Begin(itr);

		while(topFacetItr_IsValid(itr)){
			TopFacet facet = topFacetItr_GetCurr (itr);

			TopElement e1 = topFacet_GetElem(model,facet,0);
			TopElement e2 = topFacet_GetElem(model,facet,1);

			//		  cout << "Facet is adjacent to elems:  " << endl;
			//		  e1.println();
			//		  e2.println();
			//		  cout  << endl;

			countall++;

			if(topElement_IsValid(model,e1) && topElement_IsValid(model,e2)){
				countvalid++;
			}

			topFacetItr_Next(itr);
		}
		topFacetItr_Destroy (itr);

		cout << "There are initially " << countall << " facets ( " << countvalid << " adjacent to valid elements)" << endl;
	}
#endif


	cout << endl << endl;


#if 1
	{

		double starttime = FEM_Timer();
		int count = 0;

		TopFacetItr *itr = topModel_CreateFacetItr (model);
		topFacetItr_Begin(itr);

		while(topFacetItr_IsValid(itr)){
			TopFacet facet = topFacetItr_GetCurr (itr);

			TopElement e1 = topFacet_GetElem(model,facet,0);
			TopElement e2 = topFacet_GetElem(model,facet,1);

			if(topElement_IsValid(model,e1) && topElement_IsValid(model,e2)){
				count++;

#if DEBUG
				CkPrintf("Found a facet with two adjacent elements.\n");
#endif

#if DEBUG
				cout << "Inserting a cohesive element there\n" << endl;
#endif
				TopElement newCohElem = topModel_InsertCohesiveAtFacet (model, TOP_ELEMENT_COH3T3, facet);
#if DEBUG

				cout << "New Element:" << endl;
				newCohElem.println();
#endif
				CkAssert(topElement_IsValid(model, newCohElem));
#if DEBUG
				cout << "num cohesives is now:" << model->mesh->elem[TOP_ELEMENT_COH3T3].count_valid() << endl;
#endif
			}

			topFacetItr_Next(itr);
		}

		topFacetItr_Destroy (itr);


		CkAssert(model->mesh->elem[TOP_ELEMENT_COH3T3].count_valid() == count);

		


		double endtime = FEM_Timer();

		if(myId==0){
			double timesec = endtime-starttime;
			double timeusec = timesec * 1000000;

			cout << "Time Taken to insert cohesive elements in driver: " << timesec << " seconds" << endl;
			cout << "PE=0 Inserted " << count << " cohesive elements " << endl;
	
			cout << "PE=0 Time to insert one cohesive is " << timeusec / count << " us " << endl;
			
		}

	}
#endif

	cout << "PE=0 Mesh now contains " << topModel_GetNElem(model) << " elements and " << topModel_GetNNodes(model) << " nodes" << endl;

#if DEBUG
	printElems(model);
	printNodes(model);
#endif

#if 1
	ensureNoDuplicateNodes(model);
#endif

	
	myOutputNetFEM(model);




	topModel_Destroy (model);
}
