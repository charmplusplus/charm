#include "geom_util.h"

int main()
{
  cout << endl;
  cout << "----------------------------" << endl
       << " geom_util.C unit tests   " << endl;
  //       << "----------------------------" << endl<<endl;
 //  cout << "Testing duplicate removal..." << endl;
//   vector<float> v;
//   v.push_back(0.0);
//   v.push_back(0.0);
//   v.push_back(1.0);
//   v.push_back(2.0);
//   v.push_back(6.0);
//   v.push_back(6.0);
//   v.push_back(1.0);
//   v.push_back(2.0);
//   v.push_back(0.0);
//   v.push_back(0.0);

//   erase_consecutive_dups(v);

//   cout << "<";
//   for (vector<float>::iterator it = v.begin(); it!=v.end(); ++it) 
//     {
//       cout << *it << " ";
//     }
//   cout << ">";
//   cout <<  " should be <0 1 2 6 1 2 >"<< endl << endl;

   tPolygond P,Q;
  P.resize(3);
  P[0][0]=0; P[0][1]=0; 
  P[1][0]=0; P[1][1]=1;  
  P[2][0]=1; P[2][1]=0; 
  Q.resize(3);
  Q[0][0]=0.1; Q[0][1]=0.1; 
  Q[1][0]=0.1; Q[1][1]=1.0;  
  Q[2][0]=0.8; Q[2][1]=0.0; 


  cout <<  "----------------------------------" << endl;
  cout << "Testing point in polygon..." << endl;
  bool res =true;
  for(short i=0;i<3;i++)
    if (!point_in_convpoly2D(Q[i], P))
      {
	res= false;
	cout << "FAILED" << endl;
      }

  if (res)
    cout << "PASSED" << endl;
  
  cout <<  "----------------------------------" << endl;
  cout << "Testing polygon in polygon..." << endl;

 
  if ( poly_in_convpoly2D(Q, P))
    cerr << "PASSED: P is in Q\n";
  else   
    cerr << "FAILED: Polygon containment test failed\n";

  if  (poly_in_convpoly2D(P, Q))
    cerr << "FAILED: Polygon containment test failed\n";
  else
    cerr << "PASSED: Q is not in P" << endl;


  cout <<  "----------------------------------" << endl;
  cout << "Testing triangle prism intersection..." << endl;
  double tri[3][3] = {0,0.25,0.25,
		      0,0.25,0.0,
		      0.25,0.25,0.25};

  double tri2[3][3] = {0.0, 0.15, 0,
		      0.25, 0.15, 0.0,
		      0, 0.15, 0.25};

  double tri3[3][3] = {-1.0, 0.26, 1.25,
		      0.0, 0.26, 0.0,
		       1.25, 0.26, 1.25};
  double tri4[3][3] = {-1, 0.25, 0,
		      1.0, 0.25, 0.0,
		       0, 0.25, 1};

  double pri[6][3] = {0, 0.25, 0.25,
		      0, 0.25, 0,
		      0.25, 0.25, 0.25,
		      0, -0.25, 0.25,
		      0, -0.25, 0,
		      0.25, -0.25, 0.25
                      };

  vector<Vec3D> xpoly2;

  //Test tri in prism
  if (tri_prism_X(tri,pri,xpoly2)== 0.03125)
    {
      cerr << "PASSED: Triangle in prism\n";
    }
  else
    {
      cerr << "FAILED: Triangle in prism, area reported " 
	   << tri_prism_X(tri,pri,xpoly2) <<"\n";
    }

  //Test prism trace in tri
  if (tri_prism_X(tri4,pri,xpoly2)== 0.03125)
    {
      cerr << "PASSED: Prism trace in triangle\n";
    }
  else
    {
      cerr << "FAILED: Prism trace in triangle, area reported " 
	   << tri_prism_X(tri4,pri,xpoly2) <<"\n";
    }

  //Test no intersection
  if (tri_prism_X(tri3,pri,xpoly2)== 0)
    {
      cerr << "PASSED: No intersection\n";
    }
  else
    {
      cerr << "FAILED: No interesection\n";
    }

  // Test 3 point intersection 
 if (tri_prism_X(tri2,pri,xpoly2)== 0.015625)
    {
      cerr << "PASSED: Triangle intersects prism\n";
    }
  else
    {
      cerr << "FAILED: Triangle intersects prism, area reported " 
	   << tri_prism_X(tri2,pri,xpoly2) <<"\n";
    }
  return 0;
}
