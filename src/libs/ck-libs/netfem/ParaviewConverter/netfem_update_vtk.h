/* 
 *  Extend the class NetFEM_update so that we can PUP data into it,   
 *  but also have functions to output its data in VTK format
 *  
 *  Author: Isaac Dooley 03/12/05
 *
 */

#ifndef __NETFEM_UPDATE_VTK__
#define __NETFEM_UPDATE_VTK__

#include <string>
#include "charm++.h"
#include "netfem.h"
#include "charm-api.h"
#include "pup_toNetwork4.h"
#include "conv-ccs.h"
#include "netfem_data.h"




/* The available Cell types, taken from vtkCellType.h 
   Currently these are not all possible to use */

/* Linear cells */
#define VTK_EMPTY_CELL     0
#define VTK_VERTEX         1
#define VTK_POLY_VERTEX    2
#define VTK_LINE           3
#define VTK_POLY_LINE      4
#define VTK_TRIANGLE       5
#define VTK_TRIANGLE_STRIP 6
#define VTK_POLYGON        7
#define VTK_PIXEL          8
#define VTK_QUAD           9
#define VTK_TETRA         10
#define VTK_VOXEL         11
#define VTK_HEXAHEDRON    12
#define VTK_WEDGE         13
#define VTK_PYRAMID       14
#define VTK_PENTAGONAL_PRISM 15
#define VTK_HEXAGONAL_PRISM  16

  /* Quadratic, isoparametric cells */
#define VTK_QUADRATIC_EDGE       21
#define VTK_QUADRATIC_TRIANGLE   22
#define VTK_QUADRATIC_QUAD       23
#define VTK_QUADRATIC_TETRA      24
#define VTK_QUADRATIC_HEXAHEDRON 25
#define VTK_QUADRATIC_WEDGE      26
#define VTK_QUADRATIC_PYRAMID    27










class NetFEM_update_vtk : public NetFEM_update {
  
 public:
  NetFEM_update_vtk():NetFEM_update(0,0,0){} // the 0's are ok, since the data will overwritten when  PUP'ed in from the files
	
	void load(char* filename);
	void save(char* filename);
	
	void saveIndex(char* filename, char* chunkfile, int t, int num_chunks);
	
	std::string vtkFileFormat();
	std::string vtkIndexFormat(int t, int num_chunks);
	
	int guessCellType(int wid);
};

#endif
