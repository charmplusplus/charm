#include<string.h>

class IdlString {
private:
  char *s;
  char *dup(const char *const a)
  {
    char *c = new char[strlen(a)+1];
    strcpy(c, a);
    return c;
  }
public:
  IdlString(): s(0) {};
  ~IdlString() { delete [] s; }
  IdlString(const char *const a) : s(0)
  {
    s = dup(a);
  };
  IdlString(const IdlString &a) : s(0)
  {
    s = dup(a.s);
  }
  //     operator IdlString() const ()
  // 	{
  // 	    return s;
  // 	}
  operator const char*const () const  // conversion from IdlString to C-style string
  {
    return s;
  }

  IdlString& operator=(const IdlString &rhs);

  friend IdlString operator+(IdlString &lhs, const IdlString &rhs)
  {
    char *c = new char[strlen(lhs.s)+strlen(rhs.s)+1];
    strcpy(c, lhs.s);
    strcat(c, rhs.s);
//     delete [] lhs.s;
//     lhs.s = c;
//     return lhs;
    IdlString ss(c);
    delete [] c;
    return ss;
  }
  IdlString& operator+=(IdlString rhs)
  {
    *this = *this + rhs;
    return *this;
  }
//   friend ostream& operator<<(ostream& out, const IdlString& a)
//   {
//     out << a.s;
//     return out;
//   }
};


IdlString& IdlString::operator=(const IdlString &rhs)
{
  if (this != &rhs) {
    delete [] s;
    s = IdlString::dup(rhs.s);
  }
  return *this;
}

