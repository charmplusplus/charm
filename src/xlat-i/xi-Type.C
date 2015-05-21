#include "xi-Type.h"
#include "xi-Parameter.h"
#include "xi-Template.h"

namespace xi {

void FuncType::print(XStr& str) { 
  rtype->print(str);
  str << "(*" << name << ")(";
  if(params)
    params->print(str);
  str << ")";
}

void
Type::genProxyName(XStr &str,forWhom forElement)
{
  (void)str; (void)forElement;
  die("type::genProxyName called (INTERNAL ERROR)");
}
void
Type::genIndexName(XStr &str)
{
  (void)str;
  die("type::genIndexName called (INTERNAL ERROR)");
}
void
Type::genMsgProxyName(XStr &str)
{
  (void)str;
  die("type::genMsgProxyName called (INTERNAL ERROR)");
}

void NamedType::genProxyName(XStr& str, forWhom forElement)
{
  const char *prefix = forWhomStr(forElement);
  if (prefix == NULL)
      die("Unrecognized forElement type passed to NamedType::genProxyName");
  if (scope) str << scope;
  str << prefix;
  str << name;
  if (tparams) str << "<" << tparams << " >";
}


void
NamedType::print(XStr& str)
{
  if (scope) str << scope;
  str << name;
  if (tparams) str << "<"<<tparams<<" >";
}

void NamedType::genIndexName(XStr& str) { 
    if (scope) str << scope;
    str << Prefix::Index; 
    str << name;
    if (tparams) str << "<"<<tparams<<" >";
}

void NamedType::genMsgProxyName(XStr& str) { 
    if (scope) str << scope;
    str << Prefix::Message;
    str << name;
    if (tparams) str << "<"<<tparams<<" >";
}

void
PtrType::print(XStr& str)
{
  type->print(str);
  for(int i=0;i<numstars;i++)
    str << "*";
}

void
TypeList::print(XStr& str)
{
  type->print(str);
  if(next) {
    str << ", ";
    next->print(str);
  }
}

int TypeList::length(void) const
{
  if (next) return next->length()+1;
  else return 1;
}

void TypeList::genProxyNames(XStr& str, const char *prefix, const char *middle,
                             const char *suffix, const char *sep,forWhom forElement)
{
  if(type) {
    str << prefix;
    type->genProxyName(str,forElement);
    if (middle!=NULL) {
      str << middle;
      type->genProxyName(str,forElement);
    }
    str << suffix;
  }
  if(next) {
    str << sep;
    next->genProxyNames(str, prefix, middle, suffix, sep,forElement);
  }
}


}   // namespace xi
