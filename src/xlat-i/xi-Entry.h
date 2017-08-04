#ifndef _ENTRY_H
#define _ENTRY_H

#include "xi-Member.h"
#include "xi-SdagConstruct.h"
#include "xi-Template.h"
#include "CEntry.h"

using std::cerr;

namespace xi {

class Attribute {
public:
  struct Argument {
    int       value;
    char     *name;
    Argument *next;

    Argument(const char* name_, int value_, Argument *next_ = NULL)
    : value(value_), next(next_) {
      name = new char[strlen(name_) + 1];
      strcpy(name, name_);
    };

    ~Argument() {
      if (next) {
        delete next;
      }
      delete [] name;
    }
  };

  Attribute(int type, Argument* args = NULL, Attribute* next = NULL)
  : type_(type), args_(args), next_(next) { };

  ~Attribute() {
    if (args_) {
      delete args_;
    }

    if (next_) {
      delete next_;
    }
  }

  Attribute* getNext() { return next_; }
  Argument*  getArgs() { return args_; }

  int is(int type) { return (type_ == type); }

  void setNext(Attribute *next) {
    next_ = next;
  }

  Attribute* getAttribute(int attribute) {
    if (this->is(attribute)) {
      return this;
    } else if (next_) {
      return next_->getAttribute(attribute);
    } else {
      return NULL;
    }
  }

  bool hasAttribute(int attribute) {
    return (getAttribute(attribute) != NULL);
  }

  int getArgument(const char* arg, int def = -1) {
    if (args_) {
      Argument *tmp = args_;
      while (tmp) {
        if (strcmp(arg, tmp->name) == 0) {
          return tmp->value;
        } else {
          tmp = tmp->next;
        }
      }
    }

    return def;
  }
private:
  int        type_;
  Argument  *args_;
  Attribute *next_;
};


class Value;
class CStateVar;
// class SdagConstruct;
class WhenConstruct;
class WhenStatementEChecker;

/******************* Entry Point ****************/
// Entry attributes
#define STHREADED 0x01
#define SSYNC     0x02
#define SLOCKED   0x04
#define SPURE     0x10
#define SMIGRATE  0x20 //<- is magic migration constructor
#define SCREATEHERE   0x40 //<- is a create-here-if-nonexistant
#define SCREATEHOME   0x80 //<- is a create-at-home-if-nonexistant
#define SIMMEDIATE    0x100 //<- is a immediate
#define SNOKEEP       0x200
#define SNOTRACE      0x400
#define SSKIPSCHED    0x800 //<- is a message skipping charm scheduler
#define SPYTHON       0x1000
#define SINLINE       0x2000 //<- inline message
#define SIGET         0x4000 
#define SLOCAL        0x8000 //<- local message
#define SACCEL        0x10000
#define SMEM          0x20000
#define SREDUCE       0x40000 // <- reduction target
#define SAPPWORK      0x80000 // <- reduction target
#define SAGGREGATE    0x100000

/* An entry construct */
class Entry : public Member {
 public:
  XStr* genClosureTypeName;
  XStr* genClosureTypeNameProxy;
  XStr* genClosureTypeNameProxyTemp;
  int line,entryCount;
  int first_line_, last_line_;

 private:    
  Attribute *attribs;
  Type *retType;
  Value *stacksize;
  const char *pythonDoc;
    

 public:
  XStr proxyName(void);
  XStr indexName(void);

 private:
//    friend class CParsedFile;
    int hasCallMarshall;
    void genCall(XStr &dest,const XStr &preCall, bool redn_wrapper=false,
                 bool usesImplBuf = false);

    XStr epStr(bool isForRedn = false, bool templateCall = false);
    XStr epIdx(int fromProxy=1, bool isForRedn = false);
    XStr epRegFn(int fromProxy=1, bool isForRedn = false);
    XStr chareIdx(int fromProxy=1);
    void genEpIdxDecl(XStr& str);
    void genEpIdxDef(XStr& str);

    void genClosure(XStr& str, bool isDef);
    void genClosureEntryDefs(XStr& str);
    void genClosureEntryDecls(XStr& str);
    
    void genChareDecl(XStr& str);
    void genChareStaticConstructorDecl(XStr& str);
    void genChareStaticConstructorDefs(XStr& str);
    void genChareDefs(XStr& str);
    
    void genArrayDefs(XStr& str);
    void genArrayStaticConstructorDecl(XStr& str);
    void genArrayStaticConstructorDefs(XStr& str);
    void genArrayDecl(XStr& str);
    
    void genGroupDecl(XStr& str);
    void genGroupStaticConstructorDecl(XStr& str);
    void genGroupStaticConstructorDefs(XStr& str);
    void genGroupDefs(XStr& str);
    
    void genPythonDecls(XStr& str);
    void genPythonDefs(XStr& str);
    void genPythonStaticDefs(XStr& str);
    void genPythonStaticDocs(XStr& str);

    void genTramTypes();
    void genTramDefs(XStr &str);
    void genTramInstantiation(XStr &str);

    // DMK - Accel Support
    void genAccelFullParamList(XStr& str, int makeRefs);
    void genAccelFullCallList(XStr& str);
    void genAccelIndexWrapperDecl_general(XStr& str);
    void genAccelIndexWrapperDef_general(XStr& str);
    void genAccelIndexWrapperDecl_spe(XStr& str);
    void genAccelIndexWrapperDef_spe(XStr& str);
    int genAccels_spe_c_funcBodies(XStr& str);
    void genAccels_spe_c_regFuncs(XStr& str);
    void genAccels_ppe_c_regFuncs(XStr& str);

    XStr aggregatorIndexType();
    XStr dataItemType();
    XStr tramBaseType();
    XStr aggregatorType();
    XStr aggregatorGlobalType(XStr& scope);
    XStr aggregatorName();
    XStr paramType(int withDefaultVals,int withEO=0,int useConst=1);
    XStr paramComma(int withDefaultVals,int withEO=0);
    XStr eo(int withDefaultVals,int priorComma=1);
    XStr syncPreCall(void);
    XStr syncPostCall(void);
    XStr marshallMsg(void);
    XStr callThread(const XStr &procName,int prependEntryName=0);
    XStr addDummyStaticCastIfVoid(void);

    // SDAG support
    std::list<CStateVar *> estateVars;

  public:
    XStr *label;
    char *name;
    TParamList *targs;

    // SDAG support
    SdagConstruct *sdagCon;
    std::list<CStateVar *> stateVars;
    CEntry *entryPtr;
    const char *intExpr;
    ParamList *param;
    int isWhenEntry;

    void addEStateVar(CStateVar *sv) {
      estateVars.push_back(sv);
      stateVars.push_back(sv);
    }

    int tramInstanceIndex;

    // DMK - Accel Support
    ParamList* accelParam;
    XStr* accelCodeBody;
    XStr* accelCallbackName;
    void setAccelParam(ParamList* apl);
    void setAccelCodeBody(XStr* acb);
    void setAccelCallbackName(XStr* acbn);

    // DMK - Accel Support
    int accel_numScalars;
    int accel_numArrays;
    int accel_dmaList_numReadOnly;
    int accel_dmaList_numReadWrite;
    int accel_dmaList_numWriteOnly;
    int accel_dmaList_scalarNeedsWrite;

    Entry(int l, Attribute *a, Type *r, const char *n, ParamList *p, Value *sz=0, SdagConstruct *sc =0, const char *e=0, int fl=-1, int ll=-1);

    inline int hasAttribute(int attribute) {
      return (attribs != NULL) && (attribs->hasAttribute(attribute));
    }

    inline Attribute* getAttribute(int attribute) {
      return attribs ? attribs->getAttribute(attribute) : NULL;
    }

    inline void addAttribute(int attribute) {
      attribs = new Attribute(attribute, NULL, attribs);
    }

    inline void removeAttribute(int attribute) {
      Attribute *curr = attribs;
      Attribute *prev = NULL;

      while (curr) {
        if (curr->is(attribute)) {
          if (prev) {
            prev->setNext(curr->getNext());
          } else {
            attribs = curr->getNext();
          }

          delete curr;

          break;
        } else {
          prev = curr;
          curr = curr->getNext();
        }
      }
    }

    void setChare(Chare *c);
    int paramIsMarshalled(void);
    int getStackSize(void);
    int isThreaded(void);
    int isSync(void);
    int isIget(void);
    int isConstructor(void);
    bool isMigrationConstructor();
    int isExclusive(void);
    int isImmediate(void);
    int isSkipscheduler(void);
    int isInline(void);
    int isLocal(void);
    int isCreate(void);
    int isCreateHome(void);
    int isCreateHere(void);
    int isPython(void);
    int isNoTrace(void);
    int isAppWork(void);
    int isNoKeep(void);
    int isSdag(void);
    bool isTramTarget(void);

    // DMK - Accel support
    int isAccel(void);

    int isMemCritical(void);
    int isReductionTarget(void);

    void print(XStr& str);
    void check();
    void genIndexDecls(XStr& str);
    void genDecls(XStr& str);
    void genDefs(XStr& str);
    void genReg(XStr& str);
    XStr genRegEp(bool isForRedn = false);
    void preprocess();
    void preprocessSDAG();
    char *getEntryName();
    void generateEntryList(std::list<CEntry*>&, WhenConstruct *);
    void collectSdagCode(SdagCollection *sc);
    void propagateState(int);
    void lookforCEntry(CEntry *centry);
    int getLine();
    void genTramRegs(XStr& str);
    void genTramPups(XStr& scope, XStr& decls, XStr& defs);
};

// TODO(Ralf): why not simply use list<Entry*> instead?
class EntryList {
  public:
    Entry *entry;
    EntryList *next;
    EntryList(Entry *e,EntryList *elist=NULL):
    	entry(e), next(elist) {}
    void generateEntryList(std::list<CEntry*>&, WhenConstruct *);
};

}   // namespace xi

#endif  // ifndef _ENTRY_H
