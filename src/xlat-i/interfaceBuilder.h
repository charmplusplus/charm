#if !defined(CHARMXI_INTERFACE_BUILDER)
#define CHARMXI_INTERFACE_BUILDER

#include <vector>
#include <string>

#include "xi-Type.h"
#include "xi-Parameter.h"
#include "xi-Value.h"
#include "xi-Entry.h"
#include "xi-Chare.h"
#include "xi-Module.h"
#include "constructs/Constructs.h"

namespace Builder {
  struct ModuleEntity;
  struct MainChare;
  struct Chare;

  #define NUM_BUILTIN_TYPES 15

  const char* builtinTypes[NUM_BUILTIN_TYPES] =
    {"int", "long", "short", "char", "unsigned int", "unsigned long",
    "unsigned long", "unsigned long long", "unsigned short",
    "unsigned char", "long long", "float", "double", "long double",
    "void"};

  enum BUILDER_ENTRY_ATTRIBUTES
    { THREADED = STHREADED,
      SYNC = SSYNC,
      IGET = SIGET,
      EXCLUSIVE = SLOCKED,
      CREATEHERE = SCREATEHERE,
      CREATEHOME = SCREATEHOME,
      NOKEEP = SNOKEEP,
      NOTRACE = SNOTRACE,
      APPWORK = SAPPWORK,
      IMMEDIATE = SIMMEDIATE,
      EXPEDITED = SSKIPSCHED,
      INLINE = SINLINE,
      LOCAL = SLOCAL,
      MEMCRITICAL = SMEM,
      REDUCTIONTARGET = SREDUCE
    };

  enum BUILDER_CHARE_ATTRIBUTES
    { MIGRATABLE = xi::Chare::CMIGRATABLE };

  struct TypeBase {
    virtual xi::Type* generateAst() = 0;
  };

  struct Type : public TypeBase {
    char* ident;

    Type(char* ident_)
      : ident(ident_)
    { }

    virtual xi::Type* generateAst() {
      for (int i = 0; i < NUM_BUILTIN_TYPES; i++)
        if (strcmp(ident, builtinTypes[i]) == 0)
          return new xi::BuiltinType(ident);
      return new xi::NamedType(ident);
    }
  };

  struct PtrType : public TypeBase {
    Type* type;
    int numstars;

    PtrType(Type* type_, int numstars_ = 1)
      : type(type_)
      , numstars(numstars_)
    { }

    virtual xi::Type* generateAst() {
      xi::PtrType* pt = new xi::PtrType(type->generateAst());
      for (int i = 1; i < numstars; i++)
        pt->indirect();
      return pt;
    }
  };

  struct Value {
    char* val;

    Value(char* val_ = 0)
      : val(val_)
    { }

    virtual xi::Value* generateAst() {
      return new xi::Value(val);
    }
  };

  struct Parameter {
    char* name, *array;
    TypeBase* type;
    Value* val;

    Parameter(TypeBase* type_, char* name_ = 0, char* array_ = 0, Value* val_ = 0)
      : type(type_)
      , name(name_)
      , array(array_)
      , val(val_)
    { }

    // int
    // int xyz
    // int xyz[10]
    // int xyz = 0

    virtual xi::Parameter* generateAst() {
      const int lineno = 0;
      return new xi::Parameter(lineno, type->generateAst(), name, array,
                               val ? val->generateAst() : NULL);
    }
  };

  template<typename Elem, typename List>
  struct GenList {
    std::vector<Elem*> elems;

    virtual List* generateListRecur(int i) {
      if (i == elems.size() - 1)
        return new List(elems[i]->generateAst());
      else
        return new List(elems[i]->generateAst(), generateListRecur(i+1));
    }
  };

  template<typename Elem, typename List>
  struct GenListLineNo {
    std::vector<Elem*> elems;

    virtual List* generateListRecurLineNo(int i) {
      const int lineno = 0;
      if (i == elems.size() - 1)
        return new List(lineno, elems[i]->generateAst(), NULL);
      else
        return new List(lineno, elems[i]->generateAst(), generateListRecurLineNo(i+1));
    }
  };

  struct EntryType : public GenList<Parameter, xi::ParamList>  {
    void addEntryParameter(Parameter* p) {
      elems.push_back(p);
    }

    xi::ParamList* generateAst() {
      return elems.size() > 0 ?
        generateListRecur(0) :
        new xi::ParamList(new xi::Parameter(0, new xi::BuiltinType("void")));
    }
  };

  namespace SDAG {
    struct Construct {
      virtual xi::SdagConstruct* generateAst() = 0;
    };

    struct Serial : public Construct {
      char* ccode, *traceName;

      Serial(char* ccode_, char* traceName_ = 0)
        : ccode(ccode_)
        , traceName(traceName_)
      { }

      virtual xi::SdagConstruct* generateAst() {
        const int lineno = 0;
        return new xi::SerialConstruct(ccode, traceName, lineno);
      }
    };

    struct Sequence : public Construct, public GenList<Construct, xi::SListConstruct> {
      Sequence() { }
      Sequence(Construct* c) {
        elems.push_back(c);
      }

      void addConstruct(Construct* c) {
        elems.push_back(c);
      }

      virtual xi::SdagConstruct* generateAst() {
        return elems.size() > 0 ? generateListRecur(0) : new xi::SListConstruct(NULL);
      }
    };

    struct Overlap : public GenList<Construct, xi::SListConstruct> {
      Overlap() { }

      void addConstruct(Construct* c) {
        elems.push_back(c);
      }

      virtual xi::SdagConstruct* generateAst() {
        return elems.size() > 0 ? generateListRecur(0) : NULL;
      }

      virtual xi::OListConstruct* generateOlistRecur(int i) {
        if (i == elems.size() - 1)
          return new xi::OListConstruct(elems[i]->generateAst());
        else
          return new xi::OListConstruct(elems[i]->generateAst(), GenList::generateListRecur(i+1));
      }
    };

    struct SEntry : public EntryType {
      char* ident, *ref;

      SEntry(char* ident_, char* ref_ = 0)
        : ident(ident_)
        , ref(ref_)
      { }

      xi::Entry* generateAst() {
        const int lineno = 0;
        return new xi::Entry(lineno, 0,
                             new xi::BuiltinType("void"),
                             ident,
                             EntryType::generateAst(), 0, 0, ref);
      }
    };

    template<typename T>
    struct WhenTemp : public Construct, public GenList<SEntry, xi::EntryList> {
      Sequence* seq;

      void addSEntry(SEntry* c) {
        elems.push_back(c);
      }

      WhenTemp(Sequence* seq_)
        : seq(seq_)
      { }

      virtual xi::SdagConstruct* generateAst() {
        return new T(elems.size() > 0 ? generateListRecur(0) : NULL, seq->generateAst());
      }
    };

    typedef WhenTemp<xi::WhenConstruct> When;
    typedef WhenTemp<xi::CaseConstruct> Case;

    struct Else : public Construct {
      Construct* cons;

      Else(Construct* cons_)
        : cons(cons_)
      { }

      virtual xi::SdagConstruct* generateAst() {
        return new xi::ElseConstruct(cons->generateAst());
      }
    };

    struct If : public Construct {
      char* cond;
      Construct* cons;
      Else* el;

      If(char* cond_, Construct* cons_, Else* el_)
        : cond(cond_)
        , cons(cons_)
        , el(el_)
      { }

      virtual xi::SdagConstruct* generateAst() {
        return new xi::IfConstruct(new xi::IntExprConstruct(cond),
                                   cons->generateAst(),
                                   el ? el->generateAst() : NULL);
      }
    };

    struct While : public Construct {
      char* cond;
      Construct* cons;

      While(char* cond_, Construct* cons_)
        : cond(cond_)
        , cons(cons_)
      { }

      virtual xi::SdagConstruct* generateAst() {
        return new xi::WhileConstruct(new xi::IntExprConstruct(cond),
                                      cons->generateAst());
      }
    };

    struct For : public Construct {
      char* decl, *pred, *advance;
      Construct* cons;

      For(char* decl_, char* pred_, char* advance_, Construct* cons_)
        : decl(decl_)
        , pred(pred_)
        , advance(advance_)
        , cons(cons_)
      { }

      virtual xi::SdagConstruct* generateAst() {
        return new xi::ForConstruct(new xi::IntExprConstruct(decl),
                                    new xi::IntExprConstruct(pred),
                                    new xi::IntExprConstruct(advance),
                                    cons->generateAst());
      }
    };

    struct ForAll : public Construct {
      char* ident;
      char* begin, *end, *step;
      Construct* cons;

      ForAll(char* ident_, char* begin_, char* end_, char* step_, Construct* cons_)
        : ident(ident_)
        , begin(begin_)
        , end(end_)
        , step(step_)
        , cons(cons_)
      { }

      virtual xi::SdagConstruct* generateAst() {
        return new xi::ForallConstruct(new xi::SdagConstruct(xi::SIDENT, ident),
                                       new xi::IntExprConstruct(begin),
                                       new xi::IntExprConstruct(end),
                                       new xi::IntExprConstruct(step),
                                       cons->generateAst());
      }
    };
  }

  struct Entry : public EntryType {
    Type* ret;
    char *name;
    Value* stackSize;
    int eattrib;
    SDAG::Construct* sdag;

    void addSDAG(SDAG::Construct* sdag_) {
      sdag = sdag_;
    }

    Entry(Type* ret_, char* name_, Value* stackSize_ = 0, SDAG::Construct* sdag_ = 0)
      : ret(ret_)
      , name(name_)
      , stackSize(stackSize_)
      , eattrib(0)
      , sdag(sdag_)
    { }

    void addAttribute(BUILDER_ENTRY_ATTRIBUTES attribute) {
      eattrib |= attribute;
    }

    xi::Entry* generateAst() {
      const int lineno = 0;
      xi::SdagConstruct* scons = NULL;
      if (sdag) scons = new xi::SdagEntryConstruct(sdag->generateAst());
      xi::ParamList* pl = EntryType::generateAst();
      xi::Entry* entry = new xi::Entry(lineno,
                                       eattrib,
                                       ret ? ret->generateAst() : NULL,
                                       name,
                                       pl,
                                       stackSize ? stackSize->generateAst() : NULL,
                                       scons);
      if (sdag) {
        scons->con1 = new xi::SdagConstruct(xi::SIDENT, name);
        scons->setEntry(entry);
        scons->con1->setEntry(entry);
        scons->param = pl;
      }
      return entry;
    }

  };

  struct ModuleEntity : public GenListLineNo<Entry, xi::AstChildren<xi::Member> > {
    int attrib;
    std::vector<char*> baseList;
    xi::TParamList *tparams;

    ModuleEntity()
      : attrib(0)
      , tparams(NULL)
    { }

    void addEntry(Entry* et) {
      elems.push_back(et);
    }

    void addBaseType(char* b) {
      baseList.push_back(b);
    }

    void addTParam(char* name, bool builtin = false) {
      xi::Type* ty = builtin ? (xi::Type*) new xi::BuiltinType(name) : (xi::Type*) new xi::NamedType(name);
      tparams = new xi::TParamList(new xi::TParamType(ty), tparams);
    }

    xi::TypeList* generateBaseList() {
      return baseList.size() > 0 ? generateBaseListRecur(0) : NULL;
    }

    xi::TypeList* generateBaseListRecur(int i) {
      if (i == baseList.size() - 1)
        return new xi::TypeList(new xi::NamedType(baseList[i]));
      else
        return new xi::TypeList(new xi::NamedType(baseList[i]), generateBaseListRecur(i+1));
    }

    xi::AstChildren<xi::Member>* generateChildren() {
      return elems.size() > 0 ? generateListRecurLineNo(0) : NULL;
    }

    virtual int generateAttributes(int attrib) = 0;
    virtual xi::Construct* generateAst() = 0;
  };

  struct Module : public GenListLineNo<ModuleEntity, xi::ConstructList> {
    char* name;
    bool isMain;

    Module(char* _name, bool isMain_)
      : name(_name)
      , isMain(isMain_)
    { }

    void addModuleEntity(ModuleEntity* mcb) {
      elems.push_back(mcb);
    }

    xi::Module* generateAst() {
      const int lineno = 0;
      xi::Module* mod = new xi::Module(lineno, name,
                                       elems.size() > 0 ? generateListRecurLineNo(0) : NULL);
      if (isMain) mod->setMain();
      return mod;
    }
  };

  struct File : public GenListLineNo<Module, xi::AstChildren<xi::Module> > {
    File() {}

    void addModule(Module* m) {
      elems.push_back(m);
    }

    xi::AstChildren<xi::Module>* generateAst() {
      return elems.size() > 0 ? generateListRecurLineNo(0) : NULL;
    }
  };

  struct Readonly : public ModuleEntity {
    Type* type;
    char* name;
    // @todo add in DimList

    Readonly(Type* type_, char* name_)
      : ModuleEntity()
      , type(type_)
      , name(name_)
    { }

    virtual int generateAttributes(int attrib) {
      return attrib;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      return new xi::Readonly(lineno,
                              type->generateAst(),
                              name,
                              NULL);
    }
  };

  struct MainChare : public ModuleEntity {
    char* name;

    MainChare(char* _name)
      : ModuleEntity()
      , name(_name) { }

    virtual int generateAttributes(int attrib) {
      return attrib | xi::Chare::CMAINCHARE;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      return new xi::MainChare(lineno,
                               this->generateAttributes(attrib),
                               new xi::NamedType(name, tparams),
                               ModuleEntity::generateBaseList(),
                               ModuleEntity::generateChildren());
    }
  };

  struct Chare : public ModuleEntity {
    char* name;

    Chare(char* _name)
      : ModuleEntity()
      , name(_name)
    { }

    virtual int generateAttributes(int attrib) {
      return attrib | xi::Chare::CCHARE;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      return new xi::Chare(lineno,
                           this->generateAttributes(attrib),
                           new xi::NamedType(name, tparams),
                           ModuleEntity::generateBaseList(),
                           ModuleEntity::generateChildren());
    }
  };

  struct Group : public ModuleEntity {
    char* name;

    Group(char* _name)
      : ModuleEntity()
      , name(_name)
    { }

    virtual int generateAttributes(int attrib) {
      return attrib | xi::Chare::CGROUP;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      return new xi::Group(lineno,
                           this->generateAttributes(attrib),
                           new xi::NamedType(name, tparams),
                           ModuleEntity::generateBaseList(),
                           ModuleEntity::generateChildren());
    }
  };

  struct NodeGroup : public ModuleEntity {
    char* name;

    NodeGroup(char* _name)
      : ModuleEntity()
      , name(_name)
    { }

    virtual int generateAttributes(int attrib) {
      return attrib | xi::Chare::CGROUP;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      return new xi::NodeGroup(lineno,
                               this->generateAttributes(attrib),
                               new xi::NamedType(name, tparams),
                               ModuleEntity::generateBaseList(),
                               ModuleEntity::generateChildren());
    }
  };

  struct Array : public ModuleEntity {
    char* name, *index;

    Array(char* name_, char* index_)
      : ModuleEntity()
      , name(name_)
      , index(index_)
    { }

    virtual int generateAttributes(int attrib) {
      return attrib | xi::Chare::CARRAY;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      return new xi::Array(lineno,
                           this->generateAttributes(attrib),
                           new xi::NamedType(index),
                           new xi::NamedType(name, tparams),
                           ModuleEntity::generateBaseList(),
                           ModuleEntity::generateChildren());
    }
  };

  struct MessageVar {
    char *name;
    char *type;

    MessageVar *next;

    MessageVar(char *name_, char *type_)
      : name(name_), type(type_), next(NULL)
    { }

    xi::MsgVarList* generateAst() {
      xi::MsgVar *mv = new xi::MsgVar(new xi::NamedType(type), name, false, false);
      return new xi::MsgVarList(mv, next ? next->generateAst() : NULL);
    }
  };

  struct Message : public ModuleEntity {
    char *name;
    MessageVar* lst;

    Message(char *name_)
      : ModuleEntity()
      , name(name_)
      , lst(NULL)
    { }

    virtual int generateAttributes(int attrib) {
      return attrib;
    }

    void addMessageVar(char *type, char *name) {
      addMessageVar(new MessageVar(name, type));
    }

    void addMessageVar(MessageVar* nxt) {
      if (lst) lst->next = nxt;
      lst = nxt;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      xi::MsgVarList *mv = lst ? lst->generateAst() : NULL;
      return new xi::Message(lineno, new xi::NamedType(name), mv);
    }
  };

  struct InitCall : public ModuleEntity {
    char *name;
    int isNode;

    InitCall(char* name_, int isNode_)
      : ModuleEntity()
      , name(name_)
      , isNode(isNode_)
    { }

    virtual int generateAttributes(int attrib) {
      return attrib;
    }

    virtual xi::Construct* generateAst() {
      const int lineno = 0;
      return new xi::InitCall(lineno, name, isNode);
    }
  };

  struct ConsEntry : public Entry {
    ConsEntry(char* name_)
      : Entry(NULL, name_)
    { }
  };
}

#endif /* CHARMXI_INTERFACE_BUILDER */
