#include "inherit.h"
#include <stdlib.h>


readonly<CProxy_inhCoord> coordinator;
readonly<CkGroupID> id_gp,id_g1,id_g2,id_g3,id_g13;
readonly<CkArrayID> id_ap,id_a1,id_a2,id_a3,id_a13;

void inherit_moduleinit(void)
{
  coordinator=CProxy_inhCoord::ckNew();
  id_gp=CProxy_gp::ckNew(); 
  id_g1=CProxy_g1::ckNew(); 
  id_g2=CProxy_g2::ckNew(); 
  id_g3=CProxy_g3::ckNew(); 
  id_g13=CProxy_g13::ckNew();
  id_ap=CProxy_ap::ckNew(1); 
  id_a1=CProxy_a1::ckNew(1); 
  id_a2=CProxy_a2::ckNew(1); 
  id_a3=CProxy_a3::ckNew(1); 
  id_a13=CProxy_a13::ckNew(1);
}

void inherit_init(void) 
{
  ((CProxy_inhCoord)coordinator).startTest();
}

/*This integer lets us tell which method we called,
so if it's not the method that executes we'll be able to tell.
*/
enum {
  typeChare=1000,
  typeGroup=2000,
  typeArray=4000,
  genParent= 100,
  genChild1= 300,
  genChild2= 700,
  genChild3= 900,
  genChild13=390,
  methParent=  1,
  methLex   =  5,
  methInh   =  7
};

class inhCoord : public CBase_inhCoord {
  int sendCount;
  int doneCount;
  
  //Chare tests:
  void as_cp(CProxy_cp &p,int gen) {
    p.parent(typeChare+genParent+methParent);sendCount++;
    p.inhLexical(typeChare+genParent+methLex);sendCount++;
    p.inhVirtual(typeChare+gen+methInh);sendCount++;  
  }
  void as_c1(CProxy_c1 &p,int gen) {
    as_cp(p,gen);
    p.parent(typeChare+genParent+methParent);sendCount++;
    p.inhLexical(typeChare+genChild1+methLex);sendCount++;
    p.inhVirtual(typeChare+gen+methInh);sendCount++;  
  }
  void as_c2(CProxy_c2 &p,int gen) {
    as_c1(p,gen);
    p.parent(typeChare+genParent+methParent);sendCount++;
    p.inhLexical(typeChare+genChild2+methLex);sendCount++;
    p.inhVirtual(typeChare+gen+methInh);sendCount++;  
  }
  void as_c3(CProxy_c3 &p,int gen) {
    as_cp(p,gen);    
    p.parent(typeChare+genParent+methParent);sendCount++;
    p.inhLexical(typeChare+genChild3+methLex);sendCount++;
    p.inhVirtual(typeChare+gen+methInh);sendCount++;  
  }
  void as_c13(CProxy_c13 &p,int gen) {
    as_c1(p,gen); 
    as_c3(p,gen); 
    p.inhLexical(typeChare+genChild13+methLex);sendCount++;
    p.inhVirtual(typeChare+gen+methInh);sendCount++;  
  }

  //Group tests: (copy-and-paste programming at its worst!)
  int groupDest;
  void as_gp(CProxy_gp &p,int gen) {
    p[groupDest].parent(typeGroup+genParent+methParent);sendCount++;
    p[groupDest].inhLexical(typeGroup+genParent+methLex);sendCount++;
    p[groupDest].inhVirtual(typeGroup+gen+methInh);sendCount++;  
  }
  void as_g1(CProxy_g1 &p,int gen) {
    as_gp(p,gen);
    p[groupDest].parent(typeGroup+genParent+methParent);sendCount++;
    p[groupDest].inhLexical(typeGroup+genChild1+methLex);sendCount++;
    p[groupDest].inhVirtual(typeGroup+gen+methInh);sendCount++;  
  }
  void as_g2(CProxy_g2 &p,int gen) {
    as_g1(p,gen);
    p[groupDest].parent(typeGroup+genParent+methParent);sendCount++;
    p[groupDest].inhLexical(typeGroup+genChild2+methLex);sendCount++;
    p[groupDest].inhVirtual(typeGroup+gen+methInh);sendCount++;  
  }
  void as_g3(CProxy_g3 &p,int gen) {
    as_gp(p,gen);    
    p[groupDest].parent(typeGroup+genParent+methParent);sendCount++;
    p[groupDest].inhLexical(typeGroup+genChild3+methLex);sendCount++;
    p[groupDest].inhVirtual(typeGroup+gen+methInh);sendCount++;  
  }
  void as_g13(CProxy_g13 &p,int gen) {
    as_g1(p,gen); 
    as_g3(p,gen); 
    p[groupDest].inhLexical(typeGroup+genChild13+methLex);sendCount++;
    p[groupDest].inhVirtual(typeGroup+gen+methInh);sendCount++;  
  }

  //Array tests:
  void as_ap(CProxy_ap &p,int gen) {
    p[0].parent(typeArray+genParent+methParent);sendCount++;
    p[0].inhLexical(typeArray+genParent+methLex);sendCount++;
    p[0].inhVirtual(typeArray+gen+methInh);sendCount++;  
  }
  void as_a1(CProxy_a1 &p,int gen) {
    as_ap(p,gen);
    p[0].parent(typeArray+genParent+methParent);sendCount++;
    p[0].inhLexical(typeArray+genChild1+methLex);sendCount++;
    p[0].inhVirtual(typeArray+gen+methInh);sendCount++;  
  }
  void as_a2(CProxy_a2 &p,int gen) {
    as_a1(p,gen);
    p[0].parent(typeArray+genParent+methParent);sendCount++;
    p[0].inhLexical(typeArray+genChild2+methLex);sendCount++;
    p[0].inhVirtual(typeArray+gen+methInh);sendCount++;  
  }
  void as_a3(CProxy_a3 &p,int gen) {
    as_ap(p,gen);    
    p[0].parent(typeArray+genParent+methParent);sendCount++;
    p[0].inhLexical(typeArray+genChild3+methLex);sendCount++;
    p[0].inhVirtual(typeArray+gen+methInh);sendCount++;  
  }
  void as_a13(CProxy_a13 &p,int gen) {
    as_a1(p,gen); 
    as_a3(p,gen); 
    p[0].inhLexical(typeArray+genChild13+methLex);sendCount++;
    p[0].inhVirtual(typeArray+gen+methInh);sendCount++;  
  }
  
public:
  inhCoord() { groupDest=0; }
  void startTest(void) {
    sendCount=0;
    doneCount=0;  
    groupDest=(1+groupDest)%(CkNumPes());
    
    {CProxy_cp p=CProxy_cp::ckNew(); as_cp(p,genParent);}
    {CProxy_c1 p=CProxy_c1::ckNew(); as_c1(p,genChild1);}
    {CProxy_c2 p=CProxy_c2::ckNew(); as_c2(p,genChild2);}
    {CProxy_c3 p=CProxy_c3::ckNew(); as_c3(p,genChild3);}
    {CProxy_c13 p=CProxy_c13::ckNew(); as_c13(p,genChild13);}

    {CProxy_gp p(id_gp); as_gp(p,genParent);}
    {CProxy_g1 p(id_g1); as_g1(p,genChild1);}
    {CProxy_g2 p(id_g2); as_g2(p,genChild2);}
    {CProxy_g3 p(id_g3); as_g3(p,genChild3);}
    {CProxy_g13 p(id_g13); as_g13(p,genChild13);}

    {CProxy_ap p(id_ap); as_ap(p,genParent);}
    {CProxy_a1 p(id_a1); as_a1(p,genChild1);}
    {CProxy_a2 p(id_a2); as_a2(p,genChild2);}
    {CProxy_a3 p(id_a3); as_a3(p,genChild3);}
    {CProxy_a13 p(id_a13); as_a13(p,genChild13);}
  }
  void done(void) {
    if (++doneCount==sendCount)
      megatest_finish();
  }
};

/************ Called if an incorrect method is executed *********/
void badMeth(const char *cls,const char *meth,int t)
{
  CkError("ERROR! Incorrect method %s::%s called instead of %d!\n",
	  cls,meth,t);
  CkAbort("Incorrect method executed");
}

/************ Class definitions ***********/
//Declares the constructor and normal methods
#define BASIC(className,parentName,type,gen) \
public: \
	className() : parentName() {} \
	className(CkMigrateMessage *m) : parentName(m) {} \
	void inhLexical(int t) { \
		if (t!=type+gen+methLex) badMeth(#className,"inhLexical",t); \
		((CProxy_inhCoord)coordinator).done();\
	}\
	virtual void inhVirtual(int t) { \
		if (t!=type+gen+methInh) badMeth(#className,"inhVirtual",t); \
		((CProxy_inhCoord)coordinator).done();\
        }

//Declares the parent method
#define PARENT(className,type) \
public: \
	void parent(int t) { \
		if (t!=type+genParent+methParent) badMeth(#className,"parent",t); \
		((CProxy_inhCoord)coordinator).done();\
	}\

//Declares a complete child class
#define CHILD(className,type,gen) \
class className : public CBase_##className { \
	BASIC(className,CBase_##className,type,gen) \
};\

//----- chares ------
class cp : public CBase_cp { 
  BASIC(cp,CBase_cp,typeChare,genParent) 
  PARENT(cp,typeChare) 
};
CHILD(c1,typeChare,genChild1)
CHILD(c2,typeChare,genChild2)
CHILD(c3,typeChare,genChild3)
CHILD(c13,typeChare,genChild13)

//------ groups ------
class gp : public CBase_gp { 
  BASIC(gp,CBase_gp,typeGroup,genParent) 
  PARENT(gp,typeGroup) 
};
CHILD(g1,typeGroup,genChild1)
CHILD(g2,typeGroup,genChild2)
CHILD(g3,typeGroup,genChild3)
CHILD(g13,typeGroup,genChild13)

//------ arrays ------
class ap : public CBase_ap { 
  BASIC(ap,CBase_ap,typeArray,genParent) 
  PARENT(ap,typeArray) 
};
CHILD(a1,typeArray,genChild1)
CHILD(a2,typeArray,genChild2)
CHILD(a3,typeArray,genChild3)
CHILD(a13,typeArray,genChild13)

MEGATEST_REGISTER_TEST(inherit,"olawlor",0)
#include "inherit.def.h"


