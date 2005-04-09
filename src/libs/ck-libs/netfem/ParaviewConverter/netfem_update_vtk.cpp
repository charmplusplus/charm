/* 
 *  Extend the class NetFEM_update so that we can PUP data into it,   
 *  but also have functions to output its data in VTK format
 *  
 *  Author: Isaac Dooley 03/12/05
 *
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream> 
#include <string>
#include <cassert>
#include "netfem_update_vtk.h"

using namespace std;

// NetFEM doesn't tell us what type of cell we have,
// so we guess from the number of vertices it has
int NetFEM_update_vtk::guessCellType(int wid){
  if(getDim() == 2)
	switch (wid){
	case 1:
	  return VTK_VERTEX;
	case 2:
	  return VTK_LINE;
	case 3: 
	  return VTK_TRIANGLE;
	case 4:
	  return VTK_QUAD;
	case 6:
	  return VTK_QUADRATIC_TRIANGLE;
	default:
	  return VTK_POLY_VERTEX;
	}
  else if(getDim() == 3)
	switch (wid){
	case 1:
	  return VTK_VERTEX;
	case 2:
	  return VTK_LINE;
	case 3: 
	  return VTK_TRIANGLE;
	case 4:
	  return VTK_TETRA;
	case 6:
	  return VTK_WEDGE;
	case 8:
	  return VTK_HEXAHEDRON;
	case 10:
	  return VTK_QUADRATIC_TETRA;
	case 20:
	  return VTK_QUADRATIC_HEXAHEDRON;
	default:
	  return VTK_POLY_VERTEX;
	}
  else{
	printf("Couldn't guess cell type for cell with %d vertices\n", wid);
	exit(-1);
  }
}


// Load the NetFEM output file
void NetFEM_update_vtk::load(char* filename){
  // Read input file into buffer
  FILE *f=fopen(filename,"rb");
  assert(f);
  fseek(f,0,SEEK_END);
  long size=ftell(f);
  fseek(f,0,SEEK_SET);
  void *buf=malloc(size);
  assert(buf);
  assert(size==fread(buf,1,size,f));

  // PUP data from buffer
  PUP_toNetwork4_unpack p(buf);
  pup(p);

  free(buf);
  fclose(f);
}


// Save VTK data into a file
void NetFEM_update_vtk::save(char* filename){
  std::ofstream f;
  f.open(filename,ios_base::out | ios_base::trunc);
  f << vtkFileFormat();
  f.close();
}

// Generate a file that references all the files for this timestep
// The format is the XML based Parallel Unstructured Grid format used by ParaView
void NetFEM_update_vtk::saveIndex(char* filename, char* chunkfile, int timestep, int num_chunks){

  printf("extracting fields from %s\n", chunkfile);
  load(chunkfile); // read data from some chunk file so we know the current attributes, etc.

  std::ofstream f;
  f.open(filename,ios_base::out | ios_base::trunc);
  f << vtkIndexFormat(timestep, num_chunks);
  f.close();
}


std::string NetFEM_update_vtk::vtkFileFormat() {
  std::ostringstream resp;   // we'll build the buffer dynamically, instead of into a fixed length buffer    
  resp << "<?xml version=\"1.0\"?>\n";
  resp << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  resp << "<UnstructuredGrid>\n";
  
  const NetFEM_nodes &t = getNodes();
  int npoints=t.getItems();
  int dimensions = getDim();

  int total_cells =0; // Compute total number of cells
  for(int e=0;e<getElems();e++)
	total_cells += getElem(e).getItems();
  
  resp << "<Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << total_cells << "\">\n";
  
  // Print the Points
	
  resp << "<Points>\n";
  resp << "<DataArray type=\"Float64\" NumberOfComponents=\""<< 3 << "\" format=\"ascii\">\n";
  resp << setiosflags(std::ios::showpoint | std::ios::scientific) << std::setprecision(7);;
  for(int i=0; i<npoints; i++){
	const double *d = t.getField(0).getData(i);
	dimensions==1 && resp << d[0] << " 0.0 0.0" << " ";
	dimensions==2 && resp << d[0] << " " << d[1] << " 0.0 ";
	dimensions==3 && resp << d[0] << " " << d[1] << " " << d[2] << " ";
  }
  resp << "</DataArray>\n</Points>\n";
	
  // Print the Cells 
  std::ostringstream connString, offsetString, typeString;
  int offset=0;
  
  for(int e=0;e<getElems();e++){
	const NetFEM_elems &el = getElem(e);
	int ncells = el.getItems();
	int wid = el.getNodesPer();
	for(int i=0; i<ncells; i++){      
	  for(int j=0;j<wid;j++)
		connString << el.getConnData(i,j) <<  " ";
	}
	
	for(int i=0;i<ncells;i++){
	  offset += wid;
	  offsetString  << offset << " ";
	}
	
	int cell_type = guessCellType(wid);

	for(int i=0;i<ncells;i++)
	  typeString << cell_type << " ";
  }
	
  resp << "<Cells>\n";
  resp << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  resp << connString.str() << "\n";
  resp << "</DataArray>\n";
  resp << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  resp << offsetString.str() << "\n";
  resp << "</DataArray>\n";
  resp << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  resp << typeString.str() << "\n";
  resp << "</DataArray>\n";
  resp << "</Cells>\n";


  // Print Cell Attribute Fields
  const NetFEM_elems &el = getElem(0); // assume the first elem has all the needed attributes
  resp << "<CellData Scalars=\"";
  int nf=el.getFields();
  for (int fn=0;fn<nf;fn++) {
	const NetFEM_doubleField &f=el.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	resp << f.getName() << ",";
  }
  resp << "local_cell_num,cell_partition_num\">\n";
	
  for (int fn=0;fn<nf;fn++) {
	const NetFEM_doubleField &f=el.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	
	cout << "Number of Items: " << n << endl;

	resp << "<DataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\">\n";
	for(int i=0;i<n;i++){
	  for(int j=0;j<wid;j++){
		resp << f.getData(i)[j] << " ";
	  }
	  resp << "\n";
	}
	resp << "</DataArray>\n";
	  
  }
	
  // Print the chunk number data to use for coloring chunks
  int partition_num =getSource();
  resp << "<DataArray type=\"Int32\" Name=\"cell_partition_num\" format=\"ascii\" NumberOfComponents=\"1\">\n";
  for(int i=0;i<total_cells;i++)
	resp << partition_num << " ";
  resp << "</DataArray>\n";


  resp << "<DataArray type=\"Int32\" Name=\"local_cell_num\" format=\"ascii\" NumberOfComponents=\"1\">\n";
  for(int i=0;i<total_cells;i++)
	resp << i << " ";
  resp << "</DataArray>\n";


  resp << "</CellData>\n";
    
  // Print Point Attribute Fields
  resp << "<PointData Scalars=\"";
  nf=t.getFields();
  for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
	const NetFEM_doubleField &f=t.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	resp << f.getName() << ",";
  }
  resp << "point_partition_num\">\n";
  for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
	const NetFEM_doubleField &f=t.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	resp << "<DataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\">\n";
	for(int i=0;i<n;i++)
	  for(int j=0;j<wid;j++)
		resp << f.getData(i)[j] << " ";
	resp << "</DataArray>\n";
  }
  
  int total_points = t.getField(0).getItems();

  resp << "<DataArray type=\"Int32\" Name=\"point_partition_number\" format=\"ascii\" NumberOfComponents=\"1\">\n";
  for(int i=0;i<total_points;i++)
	resp << partition_num << " ";
  resp << "</DataArray>\n";
  
 
  resp << "</PointData>\n";
  
  resp << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
  
  // copy from STL string to a malloc'd c string
  std::string s = resp.str();
  return s;
}


// Generate the output data in VTK's newer XML format
std::string NetFEM_update_vtk::vtkIndexFormat(int timestep, int numChunks) {
  ostringstream resp;   // we'll build the buffer dynamically, instead of into a fixed buffer    
  
  resp << "<?xml version=\"1.0\"?>\n";
  resp << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  resp << "<PUnstructuredGrid GhostLevel=\"0\">\n\n";
  
  const NetFEM_nodes &t = getNodes();
  int npoints=t.getItems();
  const NetFEM_elems &el = getElem(0);
  int ntriangles = el.getItems();
  int dimensions = getDim();
	
  resp << "<PPoints>\n";
  resp << "<PDataArray type=\"Float64\" NumberOfComponents=\""<< 3 << "\" format=\"ascii\"/>\n";
  resp << "</PPoints>\n\n";

  resp << "<PCells>\n";
  resp << "<PDataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\"/>\n";
  resp << "<PDataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"/>\n";
  resp << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\"/>\n";
  resp << "</PCells>\n\n";
		
  // Print Cell Attribute Fields
  resp << "<PCellData Scalars=\"";
  int nf=el.getFields();
  for (int fn=0;fn<nf;fn++) {
	const NetFEM_doubleField &f=el.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	resp << f.getName() << ",";
  }
  resp << "cell_partition_num\">\n";
	
  for (int fn=0;fn<nf;fn++) {
	const NetFEM_doubleField &f=el.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	resp << "<PDataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\"/>\n";
	  
  }
	
  resp << "<PDataArray type=\"Int32\" Name=\"cell_partition_num\" format=\"ascii\" NumberOfComponents=\"1\"/>\n";
  resp << "<PDataArray type=\"Int32\" Name=\"local_cell_num\" format=\"ascii\" NumberOfComponents=\"1\"/>\n";	
  resp << "</PCellData>\n\n";
	

  // Print Point Attribute Fields
  resp << "<PPointData Scalars=\"";
  nf=t.getFields();
  for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
	const NetFEM_doubleField &f=t.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	resp << f.getName();
  }
  resp << "point_partition_num\">\n";
  for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
	const NetFEM_doubleField &f=t.getField(fn);
	int wid=f.getDoublesPerItem();
	int n=f.getItems();
	resp << "<PDataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\"/>\n";
     
  }

  resp << "<PDataArray type=\"Int32\" Name=\"point_partition_num\" format=\"ascii\" NumberOfComponents=\"1\"/>\n";

  resp << "</PPointData>\n\n";

  int numchunks;

  for(int i=0;i<numChunks;i++)
	resp << "<Piece Source=\"../" << timestep << "/" << i << ".vtu\" />\n";

  resp << "\n</PUnstructuredGrid>\n</VTKFile>\n";
    
  std::string s = resp.str();
  return s;
}
