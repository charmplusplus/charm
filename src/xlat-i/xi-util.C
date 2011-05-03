#include "xi-util.h"

namespace xi {

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

void XStr::initTo(const char *_s)
{
  len = strlen(_s);
  blklen = SZ;
  while ( len >= blklen ) {
    blklen += SZ;
  }
  s = new char[blklen];
  strcpy(s, _s);
}

XStr::XStr() {initTo("");}
XStr::XStr(const char *_s) {initTo(_s);}
XStr::XStr(const XStr &_s) {initTo(_s.get_string_const());}

XStr& XStr::operator << (int i) {
      char tmp[100]; 
      sprintf(tmp, "%d", i); 
      append(tmp); 
      return *this;
}

void XStr::line_append(const char c)
{
  XStr xs;
  for(unsigned int i=0; i<len; i++) {
    if(s[i] == '\n')
      xs << c << "\n";
    else
      xs << s[i];
  }
  delete[] s;
  initTo(xs.charstar());
}

void XStr::line_append_padding(const char c, int lineWidth)
{
  XStr xs;
  int count = 0;

  for(unsigned int i=0; i<len; i++) {
    if(s[i] == '\n'){
      // found line ending
      while(count++ < lineWidth-1)
	xs << " ";
      xs << c << "\n";
      count=0;
    } else if(s[i] == '\t') {
      // found tab, convert to 2 spaces
      xs << "  ";
      count+=2;
    } else {
      // found non-line ending
      xs << s[i];
      count++;
    }
  }
  delete[] s;
  initTo(xs.charstar());
}



void 
XStr::spew(const char*b, const char *a1, const char *a2, const char *a3, 
           const char *a4, const char *a5)
{
  using std::cout;
  int i,length=strlen(b);
  for(i=0; i<length; i++){
    switch(b[i]){
    case '\001':
      if(a1==0) {cout << "Internal Error\n"; abort();} append(a1); break;
    case '\002':
      if(a2==0) {cout << "Internal Error\n"; abort();} append(a2); break;
    case '\003':
      if(a3==0) {cout << "Internal Error\n"; abort();} append(a3); break;
    case '\004':
      if(a4==0) {cout << "Internal Error\n"; abort();} append(a4); break;
    case '\005':
      if(a5==0) {cout << "Internal Error\n"; abort();} append(a5); break;
    default:
      append(b[i]);
    }
  }
}

void XStr::replace (const char a, const char b) {
  for(unsigned int i=0; i<len; i++) {
    if (s[i] == a) s[i] = b;
  }
}

}
