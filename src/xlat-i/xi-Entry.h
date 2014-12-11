#ifndef _ENTRY_H
#define _ENTRY_H

#include "xi-Member.h"
#include "xi-SdagConstruct.h"
#include "xi-Template.h"
#include "CEntry.h"

using std::cerr;

namespace xi {

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
  int attribs;    
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
    XStr aggregatorName();
    XStr paramType(int withDefaultVals,int withEO=0,int useConst=1);
    XStr paramComma(int withDefaultVals,int withEO=0);
    XStr eo(int withDefaultVals,int priorComma=1);
    XStr syncPreCall(void);
    XStr syncPostCall(void);
    XStr marshallMsg(void);
    XStr callThread(const XStr &procName,int prependEntryName=0);

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

    Entry(int l, int a, Type *r, const char *n, ParamList *p, Value *sz=0, SdagConstruct *sc =0, const char *e=0, int fl=-1, int ll=-1);
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
    int isTramTarget(void);

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
    void genTramPups(XStr& decls, XStr& defs);
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
