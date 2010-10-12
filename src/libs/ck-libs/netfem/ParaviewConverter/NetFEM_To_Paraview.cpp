/* 
 *  Convert NetFEM files to Paraview format
 *  Run this from the directory containing "NetFEM".
 *
 *  Author: Isaac Dooley 03/12/05
 *
 */

#include <string>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <errno.h> 
#include <dirent.h> 
#include "netfem_data.h"
#include "netfem_update_vtk.h"

#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

void convert(char* from, char* to){
  NetFEM_update_vtk *u=new NetFEM_update_vtk();
  u->load(from);
  u->save(to);
  delete u;
}


void save_index(char* to, char* chunkfile, int t, int num_chunks){
  NetFEM_update_vtk *u=new NetFEM_update_vtk();
  u->saveIndex(to, chunkfile, t, num_chunks);
  delete u;
}


int main(int argc, char **argv)
{
  char infile[1024];
  char outfile[1024];
  char tfile[1024];
 
  DIR *pdir, *pdir2;
  struct dirent *pent, *pent2;

  int chunk_num;
  int total_chunks=0;

  if (pdir=opendir("NetFEM")) {
	printf("Successfully opened directory NetFEM\n");
  } else{
	printf("ERROR: couldn't open directory NetFEM\n");
	printf("Run from directory containing the NetFEM directory\n");
	exit(-1);
  }
	
  // create directory for the output vtk files
  char dirName[1024]; // temporary buffer for building filenames
  sprintf(dirName,"ParaViewData");
  mkdir(dirName,0777);
  sprintf(dirName,"ParaViewData/timesteps");
  mkdir(dirName,0777);

  int low=0, high=INT_MAX, stride=1;
  if (argc > 1) low = (int)strtol(argv[1], NULL, 10);
  if (argc > 2) high = (int)strtol(argv[2], NULL, 10);
  if (argc > 3) stride = (int)strtol(argv[3], NULL, 10);
  printf("Processing timesteps [%d:%d:%d]\n", low, high, stride);

  // process data for each timestep
  for (int timestep = low; timestep<=high; timestep += stride) {
	  
	  // open directory containing .dat files
	  sprintf(dirName,"NetFEM/%d", timestep);
          if (!(pdir2 = opendir(dirName))) break;
	  
	  // create output directory
	  sprintf(dirName,"ParaViewData/%d", timestep);
	  mkdir(dirName,0777);

	  // convert each file for this timestep
	  total_chunks=0;
	  while ((pent2=readdir(pdir2))) {
		if(strcmp(pent2->d_name,".") && strcmp(pent2->d_name,"..")) {
		  sprintf(infile, "NetFEM/%d/%s", timestep, pent2->d_name);
		  assert(sscanf(pent2->d_name, "%d.dat", &chunk_num)==1);
		  sprintf(outfile, "ParaViewData/%d/%d.vtu", timestep, chunk_num);
		  printf("Converting file %s\n",infile);
		  fflush(stdout);
		  convert(infile,outfile);
		  total_chunks++;
		}
          }
	  	  
	  // Create the index file which references 
	  // all the chunks from this timestep
	  sprintf(tfile, "ParaViewData/timesteps/step_%010d.pvtu", timestep);   // save index to here
	  sprintf(infile, "NetFEM/%d/0.dat", timestep); // the chunk file from which attributes will be extracted
	  save_index(tfile, infile, timestep, total_chunks);
	  closedir(pdir2);
  }
  
  closedir(pdir);
  printf("done\n");
  return 0;
}

