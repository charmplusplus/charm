#ifndef _PARAMETER_H
#define _PARAMETER_H

#include "xi-util.h"

namespace xi {

class Type;
class Entry;
class Value;

/**************** Parameter types & lists (for marshalling) ************/
class Parameter {
 public:
    Type *type;
    const char *name; /*The name of the variable, if any*/
    const char *given_name; /*The name of the msg in ci file, if any*/
    const char *arrLen; /*The expression for the length of the array;
    			 NULL if not an array*/
    Value *val; /*Initial value, if any*/
    int line;
    int byReference; //Fake a pass-by-reference (for efficiency)
    bool declaredReference; // Actually was declared a reference
    int conditional; //If the parameter is conditionally packed
    bool byConst;

    // DMK - Added field for accelerator options
    int accelBufferType;
    XStr* accelInstName;
    bool podType;

    friend class ParamList;
    void pup(XStr &str);
    void copyPtr(XStr &str);
    void marshallArraySizes(XStr &str);
    void marshallArrayData(XStr &str);
    void beginUnmarshall(XStr &str);
    void beginUnmarshallSDAGCall(XStr &str);
    void unmarshallArrayData(XStr &str);
    void unmarshallArrayDataSDAG(XStr &str);
    void unmarshallArrayDataSDAGCall(XStr &str);
    void pupAllValues(XStr &str);

  public:
    Entry *entry;
    Parameter(int Nline,Type *Ntype,const char *Nname=0,
    	const char *NarrLen=0,Value *Nvalue=0);
    void setConditional(int c);
    void print(XStr &str,int withDefaultValues=0,int useConst=1);
    void printAddress(XStr &str);
    void printValue(XStr &str);
    int isMessage(void) const;
    int isVoid(void) const;
    int isCkArgMsgPtr(void) const;
    int isCkMigMsgPtr(void) const;
    int isArray(void) const;
    int isConditional(void) const;
    Type *getType(void) {return type;}
    const char *getArrayLen(void) const {return arrLen;}
    const char *getGivenName(void) const {return given_name;}
    void setGivenName(const char* s) {given_name = s;}
    const char *getName(void) const {return name;}
    void printMsg(XStr& str);
    int operator==(const Parameter &parm) const;

    // DMK - Added for accelerator options
    public:
    enum {
      ACCEL_BUFFER_TYPE_UNKNOWN   = 0,
      ACCEL_BUFFER_TYPE_MIN       = 1,
      ACCEL_BUFFER_TYPE_READWRITE = 1,
      ACCEL_BUFFER_TYPE_READONLY  = 2,
      ACCEL_BUFFER_TYPE_WRITEONLY = 3,
      ACCEL_BUFFER_TYPE_MAX       = 3
    };
    void setAccelBufferType(int abt);
    int getAccelBufferType();
    void setAccelInstName(XStr* ain);
    XStr* getAccelInstName(void);
};

class ParamList {
    typedef int (Parameter::*pred_t)(void) const;
    int orEach(pred_t f);
    typedef void (Parameter::*fn_t)(XStr &str);
    void callEach(fn_t f,XStr &str);
    bool manyPointers;

  public:
    Entry *entry;
    Parameter *param;
    ParamList *next;
    ParamList(ParamList *pl);
    ParamList(Parameter *Nparam,ParamList *Nnext=NULL);
    void print(XStr &str,int withDefaultValues=0,int useConst=1);
    void printTypes(XStr &str,int withDefaultValues=0,int useConst=1);
    void printAddress(XStr &str);
    void printValue(XStr &str);
    int isNamed(void) const;
    int isBuiltin(void) const;
    int isMessage(void) const;
    const char *getArrayLen(void) const;
    int isArray(void) const;
    int isReference(void) const;
    int declaredReference(void) const;
    bool isConst(void) const;
    int isVoid(void) const;
    int isPointer(void) const;
    const char *getGivenName(void) const;
    void setGivenName(const char* s);
    const char *getName(void) const;
    int isMarshalled(void) const;
    int isCkArgMsgPtr(void) const;
    int isCkMigMsgPtr(void) const;
    int getNumStars(void) const;
    const char *getBaseName(void);
    void genMsgProxyName(XStr &str);
    void printMsg(XStr& str);
    void preprocess();
    int hasConditional();
    void marshall(XStr &str, XStr &entry);
    void beginUnmarshall(XStr &str);
    void beginUnmarshallSDAG(XStr &str);
    void beginUnmarshallSDAGCall(XStr &str, bool usesImplBuf);
    void beginRednWrapperUnmarshall(XStr &str, bool isSDAGGen);
    void unmarshall(XStr &str, int isFirst=1);
    void unmarshallSDAGCall(XStr &str, int isFirst=1);
    void unmarshallAddress(XStr &str, int isFirst=1);
    void pupAllValues(XStr &str);
    void endUnmarshall(XStr &str);
    int operator==(ParamList &plist);
    void checkParamList();
};

}   // namespace xi 

#endif // ifndef _TYPE_H
