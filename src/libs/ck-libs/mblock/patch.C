/*
A patch is a portion of a block's face--
an interface between this block and something else.

This file has routines to read blocks and patches.

Orion Sky Lawlor, olawlor@acm.org, 7/18/2001
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "charm++.h"
#include "patch.h"

static void abort(const char *why) {
  fprintf(stderr,"Fatal error reading patch: %s\n",why);
  exit(1);
}

class patchReader {
  const char *fName;
  FILE *f;
  int lineCount;
  char line[1024];//Current line in the file
  int curChar; //Current character in the line

  void abort(const char *why) {
    CkError("Fatal error reading patch description: %s\n",why);
    CkError(" while parsing line %d of file '%s'.\n",lineCount,fName);
    CkExit();
  }
public:
  patchReader(const char *fileName) {
    lineCount=0; fName=fileName; //(Only used for error messages)
    f=fopen(fileName,"r");
    if (f==NULL) {
	CkError("Couldn't open patch file '%s'!\n",fileName);
	CkExit();
    }
    nextLine();
  }
  ~patchReader() {fclose(f);}
  
  //Advance to the next non-empty, non-comment line in the file
  int nextLine(void) {
    while (NULL!=fgets(line,1024,f)) {
      curChar=0;//Start at beginning of line
      lineCount++;
      if (line[0]=='#' || line[0]=='!') continue;
      strtok(line,"!#"); //Clip off comments
      if (line[0]==0 || line[0]=='\n') continue;
      return 1; //Must be a good line
    }
    return 0;//Ran out of input file
  }
  
  //Return the next integer from the file
  int nextInt(void) {
    int ret=0,offset=0;
    if (sscanf(&line[curChar],"%d%n",&ret,&offset)<1)
      abort("couldn't parse int");
    curChar+=offset;
    return ret;
  }
  //Return the next double from the file
  double nextDouble(void) {
    double ret=0.0;
    int offset=0;
    if (sscanf(&line[curChar],"%lg%n",&ret,&offset)<1)
      abort("couldn't parse double");
    curChar+=offset;
    return ret;
  }
  //Return a block dimension from the file
  blockDim nextDim(void) {
    blockDim ret;
    for (int axis=0;axis<3;axis++)
      ret[axis]=nextInt();
    return ret;
  }
  //Return a block span from the file
  blockSpan nextSpan(void) {
    blockSpan ret;
    for (int axis=0;axis<3;axis++) {
      ret.start[axis]=nextInt(); 
      ret.end[axis]=nextInt();
    }
    return ret;
  }
};


block::block(const char *filePrefix,int blockNo)
{ 
  char fName[1024];
  
  //Read the boundary descriptions
  sprintf(fName,"%s%05d.bblk",filePrefix,blockNo);
  {
    patchReader f(fName);
    double version=f.nextDouble(); f.nextLine();
    if (version>=2.0) abort("Incompatible block version!\n");
    int blockNo=f.nextInt(); dim=f.nextDim(); f.nextLine();
    int nFaces=f.nextInt(); nPatches=f.nextInt(); f.nextLine();
    typedef patch *patchPtr;
    patches=new patchPtr[nPatches];
    int curPatch=0;
    for (int fc=0;fc<nFaces;fc++) 
    {//Read the patches for this face
      int nP=f.nextInt();f.nextLine();
      for (int pc=0;pc<nP;pc++) 
      { //Read the next patch
	int type=f.nextInt();
	blockSpan span=f.nextSpan();
	f.nextLine();
	patch *p;
	if (type==-1) { //Internal patch
	  int destBlock=f.nextInt();
	  int destPatch=f.nextInt();
	  int orient[3];
	  for (int axis=0;axis<3;axis++)
	    orient[axis]=f.nextInt();
	  f.nextLine();
	  p=new internalBCpatch(destBlock,destPatch,orient, span);
	}
	else
	  p=new externalBCpatch(span,type);
	patches[curPatch++]=p;
      } 
    }
    if (curPatch!=nPatches) abort("Didn't define all patches!");
  }
  
  //Read the mesh locations themselves
  sprintf(fName,"%s%05d.mblk",filePrefix,blockNo);
  FILE *fm=fopen(fName,"r");
  if (fm==NULL) abort("Can't open .mblk file!");
  int sizes;
  if (3!=fscanf(fm,"%d%d%d",&sizes,&sizes,&sizes)) abort("Can't parse .mblk file's header");
  locs=new vector3d[dim.getSize()];
  blockLoc i;
  BLOCKSPAN_FOR(i,blockSpan(blockLoc(0,0,0),dim)) {
    double x,y,z;
    if (3!=fscanf(fm,"%lf%lf%lf",&x,&y,&z)) abort("Can't parse .mblk file's location");
    locs[dim[i]]=vector3d(x,y,z);
  }
  fclose(fm);
}

block::~block() {
  delete[] locs;
  for (int p=0;p<nPatches;p++)
    delete patches[p];
  delete[] patches;
}

orientation::orientation(const int *codedOrient)
{
  for (int axis=0;axis<3;axis++) {
    int code=codedOrient[axis];
    flip[axis]=(code<0);
    if (flip[axis]) code=-code;
    s2d[axis]=code-1;
  }
}

patch::patch(const blockSpan &span_)
	  :span(span_) 
{ 
  flatAxis=span.getFlatAxis();
  isLow=(span.start[flatAxis]==0);
}

//Get the extents of this patch, extruded to toWidth
blockSpan patch::getExtents(const extrudeMethod &m,bool forVoxel,int dir)
{
  //Begin with our dimensions
  blockLoc s=span.start; 
  blockLoc e=span.end;

  //Extrude along our normal axis
  int w=m.toWidth*dir;
  if (isLow) w=-w;
  if (w<0) s[flatAxis]+=w;
  else     e[flatAxis]+=w;
  
  //Add the corners if needed
  if (w<0) w=-w;
  if (m.withCorners) {
    for (int axis=0;axis<3;axis++) 
    if (axis!=flatAxis) {
      s[axis]-=w;
      e[axis]+=w;
    }
  }
  
  //Convert to voxel coordinates (from node coords)
  if (forVoxel) {
    e=e-blockLoc(1,1,1);
  } else /*forNode*/ {
    if (isLow) e[flatAxis]--;
    else       s[flatAxis]++;
  }
  return blockSpan(s,e);
}


