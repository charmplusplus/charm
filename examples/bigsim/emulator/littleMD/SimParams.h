#ifndef _SIM_PARAMS_H
#define _SIM_PARAMS_H

typedef struct {
  int    steps ;
  int    n_atoms ;
  double min_x, min_y, min_z ;
  double max_x, max_y, max_z ;  // Angstroms
  int    cell_dim_x, cell_dim_y, cell_dim_z, total_cells ;
  double cutoff ;               // Angstroms
  double margin ;               // Angstroms
  double step_sz ;              // femtoseconds (=1E-15s)

  void setParamsDebug() ;
  void setParams270K() ;
  void setParams40K() ;
} SimParams ;

inline void SimParams::setParamsDebug() {
  steps = 10 ;
  n_atoms = 300 ;
  min_x = 0 ;
  min_y = 0 ;
  min_z = 0 ;
  max_x = 40 ;
  max_y = 40 ;
  max_z = 20 ;
  cutoff = 15 ;
  margin = 2 ;
  step_sz = 1.0 ;
}

inline void SimParams::setParams270K() {
  steps = 10 ;
  n_atoms = 270000 ;  // 5000 old
  min_x = 0 ;
  min_y = 0 ;
  min_z = 0 ;
  max_x = 450 ;  // 300 old
  max_y = 450 ;  // 300 old
  max_z = 450 ;  // 300 old
  cutoff = 15 ;
  margin = 2 ;
  step_sz = 1.0 ;
}

inline void SimParams::setParams40K() {
  steps = 10 ;
  n_atoms = 40000 ;  // 5000 old
  min_x = 0 ;
  min_y = 0 ;
  min_z = 0 ;
  max_x = 105 ;  // 300 old
  max_y = 105 ;  // 300 old
  max_z = 105 ;  // 300 old
  cutoff = 15 ;
  margin = 2 ;
  step_sz = 1.0 ;
}

#endif
