#include "xi-Template.h"
#include "xi-AstNode.h"
#include "xi-Chare.h"

namespace xi {

TParamList::TParamList(TParam *t, TParamList *n) : tparam(t), next(n) {}

TParamType::TParamType(Type *t) : type(t) {}
void TParamType::print(XStr& str) { type->print(str); }
void TParamType::genSpec(XStr& str) { type->print(str); }

TParamVal::TParamVal(const char *v) : val(v) {}
void TParamVal::print(XStr& str) { str << val; }
void TParamVal::genSpec(XStr& str) { str << val; }

Scope::Scope(const char* name, ConstructList* contents)
  : name_(name)
  , ConstructList(-1, NULL, contents)
{ }

void Scope::genDecls(XStr& str) {
    str << "namespace " << name_ << " {\n";
    AstChildren<Construct>::genDecls(str);
    str << "} // namespace " << name_ << "\n";
}

void Scope::genDefs(XStr& str) {
    str << "namespace " << name_ << " {\n";
    AstChildren<Construct>::genDefs(str);
    str << "} // namespace " << name_ << "\n";
}

void Scope::genReg(XStr& str) {
    str << "using namespace " << name_ << ";\n";
    AstChildren<Construct>::genReg(str);
}

void Scope::genGlobalCode(XStr scope, XStr &decls, XStr &defs) {
  scope << name_ << "::";
  AstChildren<Construct>::genGlobalCode(scope, decls, defs);
}

void Scope::print(XStr& str) {
    str << "namespace " << name_ << "{\n";
    AstChildren<Construct>::print(str);
    str << "} // namespace " << name_ << "\n";
}

void Scope::outputClosuresDecl(XStr& str) {
  str << "namespace " << name_ << " {\n";
  AstChildren<Construct>::outputClosuresDecl(str);
  str << "} // namespace " << name_ << "\n";
}

void Scope::outputClosuresDef(XStr& str) {
  str << "namespace " << name_ << " {\n";
  AstChildren<Construct>::outputClosuresDef(str);
  str << "} // namespace " << name_ << "\n";
}

UsingScope::UsingScope(const char* name, bool symbol) : name_(name), symbol_(symbol) {}
void UsingScope::genDecls(XStr& str) {
    str << "using ";
    if (!symbol_) str << "namespace ";
    str << name_ << ";\n";
}
void UsingScope::print(XStr& str) {
    str << "using ";
    if (!symbol_) str << "namespace ";
    str << name_ << ";\n";
}

void TEntity::setTemplate(Template *t) { templat = t; }
XStr TEntity::tspec(bool printDefault) const {
    XStr str; 
    if (templat) templat->genSpec(str, printDefault); 
    return str;
}
XStr TEntity::tvars(void) const {
    XStr str;
    if (templat) templat->genVars(str); 
    return str;
}

TType::TType(Type *t, Type *i) : type(t), init(i) {}

TFunc::TFunc(FuncType *t, const char *v) : type(t), init(v) {}
void TFunc::print(XStr& str) { type->print(str); if(init) str << "=" << init; }
void TFunc::genLong(XStr& str, bool printDefault){ type->print(str); if(init && printDefault) str << "=" << init; }
void TFunc::genShort(XStr& str) {str << type->getBaseName(); }

TName::TName(Type *t, const char *n, const char *v) : type(t), name(n), val(v) {}

TVarList::TVarList(TVar *v, TVarList *n) : tvar(v), next(n) {}

void
Template::outputClosuresDecl(XStr& str) {
  Chare* c = dynamic_cast<Chare*>(entity);
  if (c) str << c->closuresDecl;
}

void
Template::outputClosuresDef(XStr& str) {
  Chare* c = dynamic_cast<Chare*>(entity);
  if (c) str << c->closuresDef;
}

void
Template::setExtern(int e)
{
  Construct::setExtern(e);
  entity->setExtern(e);
}

void
Template::genVars(XStr& str)
{
  str << " < ";
  if(tspec)
    tspec->genShort(str);
  str << " > ";
}

void
Template::genSpec(XStr& str, bool printDefault)
{
  str << generateTemplateSpec(tspec, printDefault);
}

void
Template::genDecls(XStr& str)
{
  if(!external && entity) {
    entity->genDecls(str);
  }
}

void
Template::genDefs(XStr& str)
{
  if(!external && entity)
    entity->genDefs(str);
}

void
Template::genGlobalCode(XStr scope, XStr &decls, XStr &defs)
{
  if(!external && entity)
    entity->genGlobalCode(scope, decls, defs);
}

int Template::genAccels_spe_c_funcBodies(XStr& str) {
  int rtn = 0;
  if (!external && entity) { rtn += entity->genAccels_spe_c_funcBodies(str); }
  return rtn;
}

void Template::genAccels_spe_c_regFuncs(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_c_regFuncs(str); }
}

void Template::genAccels_spe_c_callInits(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_c_callInits(str); }
}

void Template::genAccels_spe_h_includes(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_h_includes(str); }
}

void Template::genAccels_spe_h_fiCountDefs(XStr& str) {
  if (!external && entity) { entity->genAccels_spe_h_fiCountDefs(str); }
}

void Template::genAccels_ppe_c_regFuncs(XStr& str) {
  if (!external && entity) { entity->genAccels_ppe_c_regFuncs(str); }
}

void Template::preprocess()
{
  if (entity)
    entity->preprocess();
}

void Template::check()
{
  if (entity)
    entity->check();
}

void
TVarList::genLong(XStr& str, bool printDefault)
{
  if(tvar)
    tvar->genLong(str, printDefault);
  if(next) {
    str << ", ";
    next->genLong(str, printDefault);
  }
}

void
TVarList::genShort(XStr& str)
{
  if(tvar)
    tvar->genShort(str);
  if(next) {
    str << ", ";
    next->genShort(str);
  }
}

void TType::genLong(XStr& str, bool printDefault)
{
  str << "class ";
  if(type)
    type->print(str);
  if(init && printDefault) {
    str << "=";
    init->print(str);
  }
}

void TType::genShort(XStr& str)
{
  if(type)
    type->print(str);
}

void TName::genLong(XStr& str, bool printDefault)
{
  if(type)
    type->print(str);
  str << " "<<name;
  if(val && printDefault) {
    str << "="<<val;
  }
}

void TName::genShort(XStr& str)
{
  str << name;
}

void TParamList::genSpec(XStr& str)
{
  if(tparam)
    tparam->genSpec(str);
  if(next) {
    str << ", ";
    next->genSpec(str);
  }
}

void
TParamList::print(XStr& str)
{
  tparam->print(str);
  if(next) {
    str << ",";
    next->print(str);
  }
}

std::string TParamList::to_string()
{
    XStr s;
    print(s);
    return s.get_string();
}



void
TType::print(XStr& str)
{
  str << "class ";
  type->print(str);
  if(init) {
    str << "=";
    init->print(str);
  }
}

void
TName::print(XStr& str)
{
  type->print(str);
  str << " "<<name;
  if(val) {
    str << "=";
    str << val;
  }
}


void
TVarList::print(XStr& str)
{
  tvar->print(str);
  if(next) {
    str << ", ";
    next->print(str);
  }
}

void
Template::print(XStr& str)
{
  if(entity)
    entity->print(str);
}

}   // namespace xi
