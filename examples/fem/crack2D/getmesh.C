#include <fstream.h>
#include <stdio.h>

int 
main(int, char**)
{
  ifstream fi;
  ofstream fo;
  fi.open("crck_bar.inp");
  if (!fi)
  {
    fprintf(stderr, "Cannot open crck_bar.inp for reading\n");
    return 1;
  }
  fo.open("crck_bar.mesh");
  if (!fo)
  {
    fprintf(stderr, "Cannot open crck_bar.mesh for writing\n");
    return 1;
  }
  int itmp, i, j, k;
  double dtmp;
  fi >> itmp >> itmp;
  fi >> dtmp;
  int nump;
  fi >> nump; // number of props
  for (i=0;i<nump;i++)
    fi >> itmp >> dtmp;
  int numnp;
  fi >> numnp; // number of nodes
  for (i=0;i<numnp;i++)
    fi >> itmp >> dtmp >> dtmp;
  int numb; // num boundary nodes
  fi >> numb;
  for(i=0;i<numb;i++)
    fi >> itmp >> itmp >> itmp >> dtmp >> dtmp;
  int numclst; // number of cohesive elements
  fi >> itmp >> numclst >> itmp >> itmp >> itmp;
  int *cnodes = new int[numclst*6];
  k = 0;
  for (i=0; i< numclst; i++)
  {
    fi >> itmp;
    for(j=0;j<6;j++)
      fi >> cnodes[k++];
    fi >> itmp;
  }
  int numlst; // number of vol elements
  fi >> itmp >> numlst >> itmp;
  int *vnodes = new int[numlst*6];
  k = 0;
  for (i=0; i< numlst; i++)
  {
    fi >> itmp;
    for(j=0;j<6;j++)
      fi >> vnodes[k++];
  }
  fo << numlst+numclst << ' ' << numnp << ' ' << 6 << endl;
  k = 0;
  for(i=0;i<numclst;i++)
  {
    for(j=0;j<6;j++)
      fo << cnodes[k++]-1 << ' ';
    fo << endl;
  }
  k = 0;
  for(i=0;i<numlst;i++)
  {
    for(j=0;j<6;j++)
      fo << vnodes[k++]-1 << ' ';
    fo << endl;
  }
  cout << "Total Nodes: " << numnp << endl;
  cout << "Total Cohesive Elements: " << numclst << endl;
  cout << "Total Volumetric Elements: " << numlst << endl;
  fi.close();
  fo.close();
  return 0;
}
