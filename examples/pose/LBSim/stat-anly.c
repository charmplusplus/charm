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
   int value;
   for(i=0;i<noObjects;i++)
	   for(j=0;j<noBuckets;j++)
		   arr[i][j]=0;
   
   fp = fopen (argv[1], "rt");  /* open the file for reading */
   while(fgets(line, 80, fp) != NULL)
   {
	 /* get a line, up to 80 chars from fp done if NULL */
   	 sscanf (line, "%d %d %d", &i, &j, &value);
         arr[i][j]=value; 
     
   }
   fclose(fp);
   
   fp = fopen (argv[2], "w");
 
   long int sum;
   double avg;

   for(j=0;j<noBuckets;j++)
	{
	 sum=0;
         for(i=0;i<noObjects;i++)
                 sum+=arr[i][j];
         avg=(double)sum/noObjects;
         fprintf(fp,"%ld %lf\n",j,avg);
        }
   fclose(fp);
 }
}        

