#ifndef __Cell_h__
#define __Cell_h__

#include "Atom.h"

struct Cell {
  int    n_atoms;
  int    x, y, z;      			// index of the cell
  double min_x, min_y, min_z;
  double max_x, max_y, max_z;   // probably unnecessary?
  Atom*  atoms;                 // array to be allocated ;

  Cell() : atoms (NULL) {}
};

#endif // __Cell_h__
