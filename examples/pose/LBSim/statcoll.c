#include<stdio.h>

FILE *fp;
long int i,j;
long int objId;
long int workStartTime;
int workFinishTime;

char line[80];

int main(int argc, char **argv)
{
 if(argc!=5)
   printf("Usage: ./fparse inputfilename outputfilename maxObjects endTime\n");
 else{
   long int noObjects=atoi(argv[3]);
   long int endTime=atoi(argv[4]);
   long int noBuckets=endTime/100;
   int arr[noObjects][noBuckets];
   for(i=0;i<noObjects;i++)
	   for(j=0;j<noBuckets;j++)
		   arr[i][j]=0;
   
   fp = fopen (argv[1], "rt");  /* open the file for reading */
   while(fgets(line, 80, fp) != NULL)
   {
	 /* get a line, up to 80 chars from fp done if NULL */
   	 sscanf (line, "%d %d %d", &objId, &workStartTime, &workFinishTime);
	 if(workStartTime/100!=workFinishTime/100) {
         arr[objId][workStartTime/100]+=(workStartTime/100+1)*100-workStartTime;
		 arr[objId][workFinishTime/100]+=workFinishTime%100;
	 }
	 else
	 arr[objId][workStartTime/100]+=(workFinishTime-workStartTime);
   
   	 //printf("%ld %lf\n",data,value);	
     
   }
   fclose(fp);
   
   fp = fopen (argv[2], "w");

   for(i=0;i<noObjects;i++)
	   for(j=0;j<noBuckets;j++)
		   fprintf(fp,"%ld %ld %ld\n",i,j,arr[i][j]);
   fclose(fp);
 }
}        

