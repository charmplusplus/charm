/*Charm++ Finite Element Framework:
C++ implementation file

This code implements fem_map and fem_assemble.
Fem_map takes a mesh and partitioning table (which maps elements
to chunks) and creates a sub-mesh for each chunk,
including the communication lists. Fem_assemble is the inverse.

The fem_map algorithm is O(n) in space and time (with n nodes,
e elements, p processors; and n>e>p^2).  The central data structure
is a bit unusual-- it's a table that maps nodes to lists of processors
(all the processors that share the node).  For any reasonable problem,
the vast majority of nodes are not shared between processors; 
this algorithm uses this to keep space and time costs low.

Memory usage for the large temporary arrays is n*sizeof(peList)
+p^2*sizeof(peList), all allocated contiguously.  Shared nodes
will result in a few independently allocated peList entries,
but shared nodes should be rare so this should not be expensive.

Note that the implementation could be significantly simplified
with judicious use of std::vector (or similar class); as 
have to do a size pass, allocate, then a copy pass.

Originally written by Orion Sky Lawlor, olawlor@acm.org, 9/28/2000
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "fem_impl.h"

/*This object maps a single node to a list of the PEs
that have a copy of it.  For the vast majority
of nodes, the list will contain exactly one element.
*/
class peList {
public:
	int pe;
	int localNo;//Local number of this node on this PE
	peList *next;
	peList() {pe=-1;next=NULL;}
	peList(int Npe,int NlocalNo) {
		pe=Npe;
		localNo=NlocalNo;
		next=NULL;
	}
	~peList() {delete next;}
	int addPE(int p,int l) {
		//Add PE p to the list with local index l,
		// if it's not there already
		if (pe==p) return 0;//Already in the list
		if (pe==-1) {pe=p;localNo=l;return 1;}
		if (next==NULL) {next=new peList(p,l);return 1;}
		else return next->addPE(p,l);
	}
	void addAlways(int p,int l) {
		//Add PE p to the list with local index l
		if (pe==-1) {pe=p;localNo=l;}
		else {
			peList *nu=new peList(p,l);
			nu->next=next;
			next=nu;
		}
	}
	int localOnPE(int p) {
		//Return this node's local number on PE p
		if (pe==p) return localNo;
		else return next->localOnPE(p);
	}
	int isEmpty(void) //Return 1 if this is an empty list 
		{return (pe==-1);}
	int length(void) {
		if (next==NULL) return isEmpty()?0:1;
		else return 1+next->length();
	}
	peList &operator[](int i) {
		if (i==0) return *this;
		else return (*next)[i-1];
	}
};



/*Create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void fem_map(const FEM_Mesh *mesh,int nchunks,int *elem2chunk,ChunkMsg **msgs)
{
//Allocate messages to return
	int c;//chunk number (to receive message)
	for (c=0;c<nchunks;c++) {
		msgs[c]=new ChunkMsg; //Ctor starts all node and element counts at zero
		msgs[c]->m.copyType(*mesh);
	}
	
//First pass, build a list of the PEs that share each node
//  (also find the local element and node counts)
	peList *nodes=new peList[mesh->node.n];
	int t;//Element type
	int e;//Element number in type
	int n;//Node number around element
	for (t=0;t<mesh->nElemTypes;t++) {
		int typeStart=mesh->nElems(t);//Number of elements before type t
		const FEM_Mesh::elemCount src=mesh->elem[t]; 
		for (e=0;e<src.n;e++) {
			c=elem2chunk[typeStart+e];
			FEM_Mesh &dest=msgs[c]->m;
			const int *srcConn=&src.conn[e*src.nodesPer];
			dest.elem[t].n++;//Found a new local element
			for (n=0;n<src.nodesPer;n++)
				if (nodes[srcConn[n]].addPE(c,dest.node.n))
					dest.node.n++;//Found a new local node
		}
		
	}

//Allocate memory for all local elements and nodes
	int *curElem=new int[nchunks]; //Next local element number to add
	int *curElem_type=new int[nchunks]; //Next element-type-local element number to add
	for (c=0;c<nchunks;c++) {
		FEM_Mesh &dest=msgs[c]->m;
		dest.node.allocUdata();
		for (t=0;t<mesh->nElemTypes;t++) {
			dest.elem[t].allocUdata();
			dest.elem[t].allocConn();
		}
		msgs[c]->elemNums=new int[dest.nElems()];
		msgs[c]->nodeNums=new int[dest.node.n];
		for (n=0;n<dest.node.n;n++) msgs[c]->nodeNums[n]=-1;
		msgs[c]->isPrimary=new int[dest.node.n];
		for (n=0;n<dest.node.n;n++) msgs[c]->isPrimary[n]=0;
		curElem[c]=0;
	}

//Second pass, add each local element and node to the local lists.
//  If we used std::vector instead of just int [], we could do this all in one pass!
	for (t=0;t<mesh->nElemTypes;t++) {
		for (c=0;c<nchunks;c++) curElem_type[c]=0;//Restart type-local count
		int typeStart=mesh->nElems(t);//Number of elements before type t
		const FEM_Mesh::elemCount &src=mesh->elem[t]; 
		const FEM_Mesh::count &nsrc=mesh->node; 
		for (e=0;e<src.n;e++) {
			c=elem2chunk[typeStart+e];
			FEM_Mesh::elemCount &dest=msgs[c]->m.elem[t];
			FEM_Mesh::count &ndest=msgs[c]->m.node;
			msgs[c]->elemNums[curElem[c]++]=typeStart+e;
			//Compute this element's global and local (type-specific) number
			int geNo=e;
			int leNo=curElem_type[c]++;
			dest.udataIs(leNo,src.udataFor(geNo));

			for (n=0;n<src.nodesPer;n++)
			{//Compute each node's local and global number
				int gnNo=src.conn[geNo*src.nodesPer+n];
				int lnNo=nodes[gnNo].localOnPE(c);
				dest.conn[leNo*dest.nodesPer+n]=lnNo;
				if (msgs[c]->nodeNums[lnNo]==-1) {
					msgs[c]->nodeNums[lnNo]=gnNo;
					ndest.udataIs(lnNo,nsrc.udataFor(gnNo));
				}
			}
		}
	}
	delete[] curElem;
	delete[] curElem_type;
	
//Find chunk comm. lists
//  (also set primary PEs)
	peList *commLists=new peList[nchunks*nchunks];
	for (n=0;n<mesh->node.n;n++) {
		int len=nodes[n].length();
		if (len==1) {//Usual case: a private node
			msgs[nodes[n].pe]->isPrimary[nodes[n].localNo]=1;
		} else if (len==0) {
			CkPrintf("FEM> Warning!  Node %d is not referenced by any element!\n",n);
		} else {/*Node is referenced by more than one processor-- 
			Add it to all the processor's comm. lists.*/
			//Make the node primary on the first processor
			msgs[nodes[n].pe]->isPrimary[nodes[n].localNo]=1;
			for (int bi=0;bi<len;bi++)
				for (int ai=0;ai<bi;ai++)
	       			{
		       			peList &a=nodes[n][ai];
		       			peList &b=nodes[n][bi];
		       			commLists[a.pe*nchunks+b.pe].addAlways(b.pe,a.localNo);
		       			commLists[b.pe*nchunks+a.pe].addAlways(a.pe,b.localNo);
		       		}
		}
	}
	delete[] nodes;
	
//Copy chunk comm. lists into comm. arrays
	for (int a=0;a<nchunks;a++) {
		int b;
		commCounts &comm=msgs[a]->comm;
		//First pass: compute communicating processors
		comm.nPes=0;
		for (b=0;b<nchunks;b++) 
			if (!commLists[a*nchunks+b].isEmpty())
			//Processor a communicates with processor b
				comm.nPes++;
		
		comm.allocate();//Now that we know nPes...
		
		//Second pass: add all shared nodes
		int curPe=0;
		for (b=0;b<nchunks;b++) 
		{
			int len=commLists[a*nchunks+b].length();
			if (len>0)
			{
				comm.peNums[curPe]=b;
				comm.numNodesPerPe[curPe]=len;
				comm.nodesPerPe[curPe]=new int[len];
				peList *l=&commLists[a*nchunks+b];
				for (int i=0;i<len;i++) {
					comm.nodesPerPe[curPe][i]=l->localNo;
					l=l->next;
				}
				curPe++;
			}
		}
	}
	delete[] commLists;
}

/****************** Assembly ******************
The inverse of fem_map: reassemble split chunks into a 
single mesh.  If nodes and elements elements aren't added 
or removed, this is straightforward; but the desired semantics
under additions and deletions is unclear to me.

For now, deleted nodes and elements leave a hole in the
global-numbered node- and element- table; and added nodes
and elements are added at the end. 
*/
FEM_Mesh *fem_assemble(int nchunks,ChunkMsg **msgs)
{
	FEM_Mesh *m=new FEM_Mesh;
	int i,t,c,e,n;

//Find the global total number of nodes and elements
	int minOld_n=1000000000,maxOld_n=0,new_n=0; //Pre-existing and newly-created nodes
	for(c=0; c<nchunks;c++) {
		const ChunkMsg *msg=msgs[c];
		for (n=0;n<msg->m.node.n;n++)
			if (msg->isPrimary[n])
			{
				int g=msg->nodeNums[n];
				if (g==-1) new_n++; //Newly-created node
				else {//pre-existing node
					if (maxOld_n<=g) maxOld_n=g+1; 
					if (minOld_n>=g) minOld_n=g; 
				}
			}
	}
	if (minOld_n>maxOld_n) minOld_n=maxOld_n;
	m->node.n=(maxOld_n-minOld_n)+new_n;
	m->node.dataPer=msgs[0]->m.node.dataPer;
	m->node.allocUdata();
	for (i=m->node.udataCount()-1;i>=0;i--)
		m->node.udata[i]=0.0;
	
	m->nElemTypes=msgs[0]->m.nElemTypes;
	int *minOld_e=new int[m->nElemTypes];
	int *maxOld_e=new int[m->nElemTypes];	
	int new_e;
	for (t=0;t<m->nElemTypes;t++) {
		minOld_e[t]=1000000000;
		maxOld_e[t]=0;
		new_e=0;
		for(c=0; c<nchunks;c++) {
			const ChunkMsg *msg=msgs[c];
			int startDex=msg->m.nElems(t);
			for (e=0;e<msg->m.elem[t].n;e++)
			{
				int g=msg->elemNums[startDex+e];
				if (g==-1) new_e++; //Newly-created element
				else {//pre-existing element
					if (maxOld_e[t]<=g) maxOld_e[t]=g+1; 
					if (minOld_e[t]>=g) minOld_e[t]=g; 
				}
			}
		}
		if (minOld_e[t]>maxOld_e[t]) minOld_e[t]=maxOld_e[t];
		m->elem[t].n=(maxOld_e[t]-minOld_e[t])+new_e;
		m->elem[t].dataPer=msgs[0]->m.elem[t].dataPer;
		m->elem[t].allocUdata();
		for (i=m->elem[t].udataCount()-1;i>=0;i--)
			m->elem[t].udata[i]=0.0;
		m->elem[t].nodesPer=msgs[0]->m.elem[t].nodesPer;
		m->elem[t].allocConn();
		for (i=m->elem[t].connCount()-1;i>=0;i--)
			m->elem[t].conn[i]=-1;
	}
	
//Now copy over the local data and connectivity into the global mesh
	new_n=0;
	for(c=0; c<nchunks;c++) {
		ChunkMsg *msg=msgs[c];
		for (n=0;n<msg->m.node.n;n++)
			if (msg->isPrimary[n])
			{
				int g=msg->nodeNums[n];
				if (g==-1) //Newly-created node-- assign a global number
					g=(maxOld_n-minOld_n)+new_n++;
				else //An existing node
					g-=minOld_n;
				
				//Copy over user data
				m->node.udataIs(g-minOld_n,msg->m.node.udataFor(n));
				msg->nodeNums[n]=g;
			}
	}
	for (t=0;t<m->nElemTypes;t++) {
		new_e=0;
		for(c=0; c<nchunks;c++) {
			const ChunkMsg *msg=msgs[c];
			int startDex=msg->m.nElems(t);
			for (e=0;e<msg->m.elem[t].n;e++)
			{
				int g=msg->elemNums[startDex+e];
				if (g==-1)//Newly-created element
					g=(maxOld_e[t]-minOld_e[t])+new_e++;
				else //An existing element
					g-=minOld_e[t];
				
				//Copy over user data
				m->elem[t].udataIs(g-minOld_e[t],msg->m.elem[t].udataFor(i));
				
				//Copy over connectivity, translating from local to global
				const int *srcConn=msg->m.elem[t].connFor(e);
				int *dstConn=m->elem[t].connFor(g-minOld_e[t]);
				for (n=0;n<msg->m.elem[t].nodesPer;n++)
					dstConn[n]=msg->nodeNums[srcConn[n]];
			}
		}
	}

	delete[] minOld_e;
	delete[] maxOld_e;
	return m;
}
