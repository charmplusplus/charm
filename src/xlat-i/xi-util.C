#include "xi-util.h"

void 
XStr::append(const char *_s) 
{
  len += strlen(_s);
  if ( len >= blklen) {
    while ( len >= blklen ) {
      blklen += SZ;
    }
    char *tmp = s;
    s = new char[blklen];
    strcpy(s, tmp);
    delete[] tmp;
  }
  strcat(s, _s);
}

void 
XStr::append(char c) 
{
  char tmp[2];
  tmp[0] = c;
  tmp[1] = '\0';
  append(tmp);
}

XStr::XStr()
{
  s = new char[SZ];
  *s = '\0';
  len = 0;
  blklen = SZ;
}

XStr::XStr(const char *_s)
{
  len = strlen(_s);
  blklen = SZ;
  while ( len >= blklen ) {
    blklen += SZ;
  }
  s = new char[blklen];
  strcpy(s, _s);
}

XStr& XStr::operator << (int i) {
      char tmp[100]; 
      sprintf(tmp, "%d", i); 
      append(tmp); 
      return *this;
};

void 
XStr::spew(const char*b, const char *a1, const char *a2, const char *a3, 
           const char *a4, const char *a5)
{
  int i;
  for(i=0; i<strlen(b); i++){
    switch(b[i]){
    case '\01':
      if(a1==0) {cout << "Internal Error\n"; abort();} append(a1); break;
    case '\02':
      if(a2==0) {cout << "Internal Error\n"; abort();} append(a2); break;
    case '\03':
      if(a3==0) {cout << "Internal Error\n"; abort();} append(a3); break;
    case '\04':
      if(a4==0) {cout << "Internal Error\n"; abort();} append(a4); break;
    case '\05':
      if(a5==0) {cout << "Internal Error\n"; abort();} append(a5); break;
    default:
      append(b[i]);
    }
  }
}

