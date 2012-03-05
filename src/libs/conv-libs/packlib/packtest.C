#include <iostream>
#include "PackLib.h"

main()
{
  using namespace std;

  Packer* p = new Packer;

  const char a_i = 'a';
  const unsigned char b_i = 'b';
  const int c_i = 123;
  const unsigned int d_i = 456;
  const long e_i = 789;
  const unsigned long f_i = 101112;
  const float g_i = 0.123;
  const double h_i = 0.456;
  const int i_i[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  const double j_i[10] = { .1, .2, .3, .4, .5, .6, .7, .8, .9, .10 };

  p->pack(a_i);
  p->pack(b_i);
  p->pack(c_i);
  p->pack(d_i);
  p->pack(e_i);
  p->pack(f_i);
  p->pack(g_i);
  p->pack(h_i);
  p->pack(i_i,10);
  p->pack(j_i,10);

  int bufsz = p->buffer_size();

  cout << "Allocating buffer of " << bufsz << " bytes" << endl;
  char* buffer = new char[bufsz];
  p->fill_buffer(buffer,bufsz);

  cout << "Unpacking buffer" << endl;
  Unpacker *up = new Unpacker(static_cast<void*>(buffer));

  char a_o;
  unsigned char b_o;
  int c_o;
  unsigned int d_o;
  long e_o;
  unsigned long f_o;
  float g_o;
  double h_o;
  int i_o[10];
  double j_o[10];
  int i;
  bool ok = true;

  up->unpack(&a_o);
  if (a_i!=a_o) {
    cout << "a_i != a_o " << a_i << " " << a_o << endl;
    ok = false;
  }
  up->unpack(&b_o);
  if (b_i!=b_o) {
    cout << "b_i != b_o " << b_i << " " << b_o << endl;
    ok = false;
  }
    
  up->unpack(&c_o);
  if (c_i!=c_o) {
    cout << "c_i != c_o " << c_i << " " << c_o << endl;
    ok = false;
  }
  up->unpack(&d_o);
  if (d_i!=d_o) {
    cout << "d_i != d_o " << d_i << " " << d_o << endl;
    ok = false;
  }
  up->unpack(&e_o);
  if (e_i!=e_o) {
    cout << "e_i != e_o " << e_i << " " << e_o << endl;
    ok = false;
  }
  up->unpack(&f_o);
  if (f_i!=f_o) {
    cout << "f_i != f_o " << f_i << " " << f_o << endl;
    ok = false;
  }
  up->unpack(&g_o);
  if (g_i!=g_o) {
    cout << "g_i != g_o " << g_i << " " << g_o << endl;
    ok = false;
  }
  up->unpack(&h_o);
  if (h_i!=h_o) {
    cout << "h_i != h_o " << h_i << " " << h_o << endl;
    ok = false;
  }
  up->unpack(i_o,10);
  for(i=0; i < 10; i++)
    if (i_i[i]!=i_o[i]) {
      cout << "i_i[" << i << "] != i_o[" << i << "] " 
	   << i_i[i] << " " << i_o[i] << endl;
      ok = false;
    }
  up->unpack(j_o,10);
  for(i=0; i < 10; i++)
    if (j_i[i]!=j_o[i]) {
      cout << "j_i[" << i << "] != j_o[" << i << "] " 
	   << j_i[i] << " " << j_o[i] << endl;
      ok = false;
    }
  if (ok) {
    cout << "Tested OK!" << endl;
    exit(0);
  } else { 
    cout << "Test failed!" << endl;
    exit(1);
  }
}
    

