#ifndef __Atom_h__
#define __Atom_h__

struct Atom {
  double m;              // Mass in AMU (= 1.66E-27 kg)
  double q;              // Charge in units of elemental charge ( = 1.602E-19C)
  double x, y, z;        // Position in Angstroms ( = 1E-10 m)
  double vx, vy, vz;     // Velocity in angstroms/fs ( = 1E5m/s)
  double vhx, vhy, vhz;  // Half-step velocity, used for Verlet integration
  double fx, fy, fz;     // Force in newtons
};

#endif 
