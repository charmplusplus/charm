/*
Simple text-based client for NetFEM data.

Orion Sky Lawlor, olawlor@acm.org, 11/2/2001
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ccs-client.h"
#include "pup.h"
#include "pup_toNetwork4.h"
#include "conv-config.h"
#include "netfem_data.h"

//Print statistics by running down the rows of this data
void printStats(const double *data,int rows,int cols)
{
	double *max=new double[cols];
	double *min=new double[cols];
	double *sum=new double[cols];
	int c;
	for (c=0;c<cols;c++) {
		max[c]=-1.0e30;
		min[c]=1.0e30;
		sum[c]=0.0;
	}
	for (int r=0;r<rows;r++) {
		for (c=0;c<cols;c++) {
			double v=data[r*cols+c];
			if (v<min[c]) min[c]=v;
			if (v>max[c]) max[c]=v;
			sum[c]+=v;
		}
	}
	printf(" Summary statistics:\n");
	for (c=0;c<cols;c++) {
		printf("   [%d]: min=%.3g, max=%.3g, ave=%.3g\n",
			c,min[c],max[c],sum[c]/rows);
	}
	delete[] max; delete[] min; delete[] sum;
	
}

void print(const NetFEM_item &t) {
	printf("%d items, %d fields\n",t.getItems(),t.getFields());
	int nf=t.getFields();
	for (int fn=0;fn<nf;fn++) {
		const NetFEM_item::doubleField &f=t.getField(fn);
		int wid=f.getDoublesPerItem();
		int n=f.getItems();
		printf(" Field '%s': %d x %d  (%s)\n",
			f.getName(),n,wid,
			f.getSpatial()?"spatial":"list");
		//Compute some statistics on the data
		printStats(f.getData(0),n,wid);
		printf("\n");
	}
}


void print(const NetFEM_update &u) {
	printf("------- Timestep %d, Chunk %d ----------------\n",
		u.getTimestep(),u.getChunk());
	printf("Nodes: ---------------\n");
	print(u.getNodes());
	for (int i=0;i<u.getElems();i++)
	{
		printf("Element type %d: ---------------\n",i);
		print(u.getElem(i));
	}
}


int main(int argc,char *argv[]) {
	if (argc<3) {
		fprintf(stderr,"Usage: %s <host> <port> ...\n",argv[0]);
		exit(1);
	}
	const char *host=argv[1];
	int port=atoi(argv[2]);
	CcsServer svr;
	printf("Connecting to %s:%d\n",host,port);
	CcsConnect(&svr,host,port,NULL);

	int totNodes=0,totElem=0;
	int pe,nPe=CcsNumPes(&svr);
	for (pe=0;pe<nPe;pe++) {
		//Ask for and get the current mesh
		printf("Sending request to PE %d...\n",pe);
		CcsSendRequest(&svr,"NetFEM_current",pe,0,NULL);
		char *reply; unsigned int replySize;
		printf("Receiving response from PE %d...\n",pe);
		CcsRecvResponseMsg(&svr,&replySize,&reply,60);
		
		//Unpack the response
		printf("Unpacking %d-byte response:\n",(int)replySize);
		NetFEM_update u(0,0);
		{PUP_toNetwork4_unpack p(reply); u.pup(p);}
		
		totNodes+=u.getNodes().getItems();
		totElem+=u.getElem(0).getItems();		

		print(u);
		//abort();
	}
}




