#include "xi-AstNode.h"
#include "xi-Module.h"
#include "xi-Member.h"

#include <algorithm>

using std::list;
using std::for_each;

/**
   Apply fn_ on each Construct in the list l, passing it arg as
   the target. If between_ is passed, do that to arg between each
   element.
 */
template<typename T, typename U, typename A>
class perElemGenC
{
  void (U::*fn)(A);
  void (*between)(A);
  A arg;

 public:
  perElemGenC(list<T*> &l,
         A arg_,
         void (U::*fn_)(A),
         void (*between_)(A) = NULL)
  : fn(fn_), between(between_), arg(arg_)
  {
    for_each(l.begin(), l.end(), *this);
  }
  void operator()(T* m)
  {
    if (m)
    {
      (m->*fn)(arg);
      if (between)
        between(arg);
    }
  }
};

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A& arg_, void (U::*fn_)(A&),
// Sun Studio 7 (C++ compiler version 5.4) can't handle this
//		void (*between_)(A&) = NULL)
		void (*between_)(A&))
{
    perElemGenC<T, U, A&>(l, arg_, fn_, between_);
}

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A& arg_, void (U::*fn_)(A&))
{
    perElemGenC<T, U, A&>(l, arg_, fn_, NULL);
}

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A* arg_, void (U::*fn_)(A*),
// See above
//		void (*between_)(A*) = NULL)
		void (*between_)(A*))
{
    perElemGenC<T, U, A*>(l, arg_, fn_, between_);
}

template<typename T, typename U, typename A>
void perElemGen(list<T*> &l, A* arg_, void (U::*fn_)(A*))
{
    perElemGenC<T, U, A*>(l, arg_, fn_, NULL);
}

/**
   Apply fn_ on each non-NULL element in the list l.
   If between_ is passed, do that between each element.
 */
template<typename T, typename U>
class perElemC
{
    void (U::*fn)();
public:
    perElemC(list<T*> &l,
	       void (U::*fn_)())
	: fn(fn_)
	{
	    for_each(l.begin(), l.end(), *this);
	}
    void operator()(T* m)
	{
	    if (m) {
		(m->*fn)();
	    }
	}
};

template<typename T, typename U>
void perElem(list<T*> &l, void (U::*fn_)())
{
    perElemC<T, U>(l, fn_);
}

namespace xi {

void newLine(XStr &str)
{
  str << endx;
}

template <typename Child>
void AstChildren<Child>::push_back(Child *m)
{
  children.push_back(m);
}

template <typename Child>
void AstChildren<Child>::print(XStr& str)
{
  perElemGen(children, str, &Child::print);
}

template <typename Child>
void AstChildren<Child>::preprocess()
{
  perElem(children, &Child::preprocess);
}

template <typename Child>
void AstChildren<Child>::check() {
  perElem(children, &Child::check);
}

template <typename Child>
void AstChildren<Child>::genDecls(XStr& str)
{
  perElemGen(children, str, &Child::genDecls, newLine);
}

template <typename Child>
void AstChildren<Child>::genClosureEntryDecls(XStr& str)
{
  perElemGen(children, str, &Child::genClosureEntryDecls, newLine);
}

template <typename Child>
void AstChildren<Child>::genClosureEntryDefs(XStr& str)
{
  perElemGen(children, str, &Child::genClosureEntryDefs, newLine);
}

template <typename Child>
void AstChildren<Child>::outputClosuresDecl(XStr& str)
{
  perElemGen(children, str, &Child::outputClosuresDecl, newLine);
}

template <typename Child>
void AstChildren<Child>::outputClosuresDef(XStr& str)
{
  perElemGen(children, str, &Child::outputClosuresDef, newLine);
}

template <typename Child>
void AstChildren<Child>::genDefs(XStr& str)
{
  perElemGen(children, str, &Child::genDefs, newLine);
}

template <typename Child>
void AstChildren<Child>::genReg(XStr& str)
{
  perElemGen(children, str, &Child::genReg, newLine);
}

template <typename Child>
template <typename T>
void AstChildren<Child>::recurse(T t, void (Child::*fn)(T))
{
  perElemGen(children, t, fn);
}

template <typename Child>
void AstChildren<Child>::recursev(void (Child::*fn)())
{
  perElem(children, fn);
}

template <typename Child>
void AstChildren<Child>::genGlobalCode(XStr scope, XStr &decls, XStr &defs)
{
  for (typename list<Child*>::iterator i = children.begin(); i != children.end(); ++i) {
    if (*i) {
      (*i)->genGlobalCode(scope, decls, defs);
    }
  }
}

template <typename Child>
void AstChildren<Child>::printChareNames()
{
  perElem(children, &Child::printChareNames);
}

template <typename Child>
int AstChildren<Child>::genAccels_spe_c_funcBodies(XStr& str) {
  int rtn = 0;
  for (typename list<Child*>::iterator i = children.begin(); i != children.end(); ++i) {
	if (*i) {
	  rtn += (*i)->genAccels_spe_c_funcBodies(str);
    }
  }
  return rtn;
}

template <typename Child>
void AstChildren<Child>::genAccels_spe_c_regFuncs(XStr& str) {
  perElemGen(children, str, &Child::genAccels_spe_c_regFuncs);
}

template <typename Child>
void AstChildren<Child>::genAccels_spe_c_callInits(XStr& str) {
  perElemGen(children, str, &Child::genAccels_spe_c_callInits);
}

template <typename Child>
void AstChildren<Child>::genAccels_spe_h_includes(XStr& str) {
  perElemGen(children, str, &Child::genAccels_spe_h_includes);
}

template <typename Child>
void AstChildren<Child>::genAccels_spe_h_fiCountDefs(XStr& str) {
  perElemGen(children, str, &Child::genAccels_spe_h_fiCountDefs);
}

template <typename Child>
void AstChildren<Child>::genAccels_ppe_c_regFuncs(XStr& str) {
  perElemGen(children, str, &Child::genAccels_ppe_c_regFuncs);
}

// Explicit instantiation because of the cross-references from the driver and the grammar
template class AstChildren<Module>;
template class AstChildren<Member>;
template class AstChildren<Construct>;
template void AstChildren<Construct>::recurse<int&>(int&, void (Construct::*)(int&));
template void AstChildren<Construct>::recurse<Module*>(Module*, void (Construct::*)(Module*));
template void AstChildren<Member>::recurse<Chare*>(Chare*, void (Member::*)(Chare*));
template void AstChildren<Member>::recurse<XStr&>(XStr&, void (Member::*)(XStr&));
template void AstChildren<Member>::recurse<SdagCollection*>(SdagCollection*, void (Member::*)(SdagCollection*));
template void AstChildren<Member>::recurse<CEntry*>(CEntry*, void (Member::*)(CEntry*));
template void AstChildren<Module>::recurse<const char *>(const char*, void (Module::*)(const char*));
template void AstChildren<Member>::recurse<WhenStatementEChecker*>(WhenStatementEChecker*, void (Member::*)(WhenStatementEChecker*));

} // namespace xi
