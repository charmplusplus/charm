#include "xi-Value.h"

namespace xi {

Value::Value(const char *s)
{
  factor = 1;
  val = s;
  if (val == 0 || strlen(val) == 0) return;
  char *v = new char[strlen(val)+5];
  strcpy(v, val);
  int pos = strlen(v)-1;
  if (v[pos] == 'K' || v[pos] == 'k') {
    v[pos] = '\0';
    factor = 1024;
  }
  if (v[pos]=='M' || v[pos]=='m') {
    v[pos] = '\0';
    factor = 1024*1024;
  }
  val=v;
}

int Value::getIntVal(void)
{
  if(val==0 || strlen(val)==0) return 0;
  return (atoi((const char *)val)*factor);
}

void Value::print(XStr& str) { str << val; }

ValueList::ValueList(Value* v, ValueList* n) : val(v), next(n) {}

void ValueList::print(XStr& str) {
  if(val) {
    str << "["; val->print(str); str << "]";
  }
  if(next)
    next->print(str);
}

void ValueList::printValue(XStr& str) {
  if(val) {
    val->print(str);
  }
  if(next) {
    die("Unsupported type");
  }
}

void ValueList::printValueProduct(XStr& str) {
  if (!val)
    die("Must have a value for an array dimension");

  str << "("; val->print(str); str << ")";
  if (next) {
    str << " * ";
    next->printValueProduct(str);
  }
}

void ValueList::printZeros(XStr& str) {
  str << "[0]";
  if (next)
    next->printZeros(str);
}

}   // namespace xi
