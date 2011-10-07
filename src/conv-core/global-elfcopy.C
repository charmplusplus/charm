/*  Library to manage the program's global and static varaibles manually
 *  by copying them during context switch.
 *
 *
 *  Developed by Gengbin Zheng (gzheng@uiuc.edu) 12/06
 *
 */

#include "converse.h"
#include "cklists.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <errno.h>

#include "charm.h"
#include "pup.h"

#if CMK_HAS_ELF_H
#include <elf.h>

#define DEBUG_GOT_MANAGER 0

#if !CMK_SHARED_VARS_UNAVAILABLE
#  error "Global-elfcopy won't work properly under smp version: -copyglobals disabled"
#endif

CpvDeclare(int, CmiPICMethod);

#if CMK_AMD64 || CMK_CRAYXT
typedef Elf64_Addr    ELFXX_TYPE_Addr;
typedef Elf64_Dyn     ELFXX_TYPE_Dyn;
typedef Elf64_Rela    ELFXX_TYPE_Rel;
typedef Elf64_Sym     ELFXX_TYPE_Sym;
#define ELFXX_R_TYPE   ELF64_R_TYPE
#define ELFXX_R_SYM    ELF64_R_SYM
#define ELFXX_ST_TYPE  ELF64_ST_TYPE
#define CMK_DT_REL     DT_RELA
#define CMK_DT_RELSZ   DT_RELASZ
#define is_elf_global(x)  ((x) == R_X86_64_GLOB_DAT)
#elif CMK_IA64
#error "NOT SUPPORTED"
#else
typedef Elf32_Addr    ELFXX_TYPE_Addr;
typedef Elf32_Dyn     ELFXX_TYPE_Dyn;
typedef Elf32_Rel     ELFXX_TYPE_Rel;
typedef Elf32_Sym     ELFXX_TYPE_Sym;
#define ELFXX_R_TYPE   ELF32_R_TYPE
#define ELFXX_R_SYM    ELF32_R_SYM
#define ELFXX_ST_TYPE  ELF32_ST_TYPE
#define CMK_DT_REL     DT_REL
#define CMK_DT_RELSZ   DT_RELSZ
#define is_elf_global(x)  ((x) == R_386_GLOB_DAT)
#endif

extern ELFXX_TYPE_Dyn _DYNAMIC[];      //The Dynamic section table pointer

/****************** Global Variable Understanding *********************/
/**
 Keeps a list of global variables.
*/
class CtgGlobalList
{
  int datalen; ///< Number of bytes in the table of global data.
  struct CtgRec {
    void *got; ///< Points to our entry in the GOT.
    int off; ///< Our byte offset into the table of global data.
    int size;
    CtgRec() {got=NULL;}
    CtgRec(void *got_,int off_,int size_) :got(got_), off(off_), size(size_) {}
  };
  CkVec<CtgRec> rec;
  int nRec;
public:
  /**
   Analyze the current set of global variables, determine 
   which are user globals and which are system globals, 
   and store the list of user globals. 
  */
  CtgGlobalList();
  
  /// Return the number of bytes needed to store our global data.
  inline int getSize(void) const {return datalen;}
  
  /// Copy the current set of global data into this set,
  ///   which must be getSize() bytes.
  void read(void *datav) const;
  
  /// copy data to globals
  inline void install(void *datav) const {
    char *data=(char *)datav;
    for (int i=0;i<nRec;i++) {
      memcpy((void *)rec[i].got, data+rec[i].off, rec[i].size);
    }
  }
  
  inline void install_var(void *datav, void *ptr) const {
    char *data=(char *)datav;
    for (int i=0;i<nRec;i++) {
      long offset = (char*)ptr-(char *)rec[i].got;
      if (offset >= 0 && offset < rec[i].size) {
        memcpy((void *)rec[i].got, data+rec[i].off, rec[i].size);
        break;
      }
    }
  }

  void read_var(void *datav, void *ptr) const;
  
private:
  /* Return 1 if this is the name of a user global variable;
     0 if this is the name of some other (Charm or system)
     global variable. */
  int isUserSymbol(const char *name);
};

int CtgGlobalList::isUserSymbol(const char *name) {
    // return 1;
    if((strncmp("_", name, 1) == 0) || (strncmp("Cpv_", name, 4) == 0)
       || (strncmp("Csv_", name, 4) == 0) || (strncmp("Ctv_", name, 4) == 0)
       || (strncmp("Bnv_", name, 4) == 0) || (strncmp("Bpv_", name, 4) == 0)
       || (strncmp("ckout", name, 5) == 0) || (strncmp("stdout", name, 6) == 0)
       || (strncmp("environ", name, 7) == 0)
       || (strncmp("stderr", name, 6) == 0) || (strncmp("stdin", name, 5) == 0))
        return 0;
    
    return 1;
}

void CtgGlobalList::read(void *datav) const {
    char *data=(char *)datav;
    for (int i=0;i<nRec;i++) {
      memcpy(data+rec[i].off, (void *)rec[i].got, rec[i].size);
    }
}

void CtgGlobalList::read_var(void *datav, void *ptr) const {
    char *data=(char *)datav;
    for (int i=0;i<nRec;i++) {
      long offset = (char*)ptr-(char *)rec[i].got;
      if (offset >= 0 && offset < rec[i].size) {
        memcpy(data+rec[i].off, (void *)rec[i].got, rec[i].size);
        break;
      }
    }
}

CkVec<char *>  _namelist;
static int loaded = 0;

extern "C" int lookup_obj_sym(char *name, unsigned long *val, int *size);

// read from a file called "globals"
static void readGlobals()
{
    if (loaded) return;
    const char *fname = "globals";
printf("Loading globals from file \"%s\" ... \n", fname);
    FILE *gf = fopen(fname, "r");
    if (gf == NULL) {
      CmiAbort("Failed to load globals, file may not exist!");
    }
    while (!feof(gf)) 
    {
      char name[1024];
      fscanf(gf, "%s\n", name);
      _namelist.push_back(strdup(name));
    }
    fclose(gf);
    loaded = 1;
}


/**
   Analyze the current set of global variables, determine 
   which are user globals and which are system globals, 
   and store the list of user globals. 
 */
CtgGlobalList::CtgGlobalList() {
    datalen=0;
    nRec=0;
    
    int count;
    for (count = 0; count < _namelist.size(); count ++) 
    {
	unsigned long addr;
        int size;
	if (0 > lookup_obj_sym(_namelist[count], &addr, &size)) {
		fprintf(stderr, "%s: no such symbol\n", _namelist[count]);
		continue;
	}
        void *ptr = (void *)addr;
        int gSize = ALIGN8(size);
		    
//#if DEBUG_GOT_MANAGER
            printf("   -> %s is a user global, of size %d, at %p\n",
	      _namelist[count], size, ptr);
//#endif
		    
	rec.push_back(CtgRec(ptr,datalen,size));
	datalen+=gSize;
    }

    nRec=rec.size();
    
#if DEBUG_GOT_MANAGER   
    printf("relt has %d entries, %d of which are user globals\n\n", 
    	relt_size,nRec);
#endif
}

/****************** Global Variable Storage and Swapping *********************/
CpvStaticDeclare(CtgGlobals,_curCtg);

struct CtgGlobalStruct {
public:
    /* This is set when our data is pointed to by the current GOT */
    int installed;
    int inited;

    /* Pointer to our global data segment. */
    void *data_seg;  
    int seg_size; /* size in bytes of data segment */
    
    void allocate(int size) {
      seg_size=size;
        /* global data segment need to be isomalloc */
      if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
        data_seg=CmiIsomalloc(seg_size);
      else
        data_seg=malloc(seg_size);
      inited = 0;
    }
    
    CtgGlobalStruct(void) {
      installed=0;
      data_seg=0;
    }
    ~CtgGlobalStruct() {
      if (data_seg) {
        free(data_seg);
      }
    }
    
    void pup(PUP::er &p);
};

void CtgGlobalStruct::pup(PUP::er &p) {
    p | seg_size;
        /* global data segment need to be isomalloc pupped */
    if (CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
      //CmiIsomallocPup(&p, &data_seg);
      pub_bytes(&p, &data_seg,sizeof(char*));
    else {
      if (p.isUnpacking()) allocate(seg_size);
      p((char *)data_seg, seg_size);
    }
}

/// Singleton object describing our global variables:
static CtgGlobalList *_ctgList=NULL;
/// Singleton object describing the original values for the globals.
static CtgGlobalStruct *_ctgListGlobals=NULL;

extern "C" int init_symtab(char *exename);

/** Initialize the globals support (called on each processor). */
void CtgInit(void) {
        CpvInitialize(int, CmiPICMethod);
        CpvAccess(CmiPICMethod) = 2;
	CpvInitialize(CtgGlobal,_curCtg);
	
	if (!_ctgList) 
	{
	/*
	  First call on this node: parse out our globals:
	*/
                readGlobals();
		init_symtab(CkGetArgv()[0]);

		CtgGlobalList *l=new CtgGlobalList;
		CtgGlobalStruct *g=new CtgGlobalStruct;
		if (CmiMyNode()==0) {
			CmiPrintf("CHARM> -copyglobals enabled\n");
		}
		
		g->allocate(l->getSize());
		l->read(g->data_seg);
		l->install(g->data_seg);
		_ctgList=l;
		_ctgListGlobals=g;
	}
        /* CmiNodeAllBarrier();  no smp anyway */
	
	CpvAccess(_curCtg)=_ctgListGlobals;
}

/** Copy the current globals into this new set */
CtgGlobals CtgCreate(void) {
	CtgGlobalStruct *g=new CtgGlobalStruct;
	g->allocate(_ctgList->getSize());
	_ctgList->read(g->data_seg);
	return g;
}

/** PUP this (not currently installed) globals set */
CtgGlobals CtgPup(pup_er pv, CtgGlobals g) {
	PUP::er *p=(PUP::er *)pv;
	if (p->isUnpacking()) g=new CtgGlobalStruct;
	if (g->installed) 
		CmiAbort("CtgPup called on currently installed globals!\n");
	g->pup(*p);
	if (g->seg_size!=_ctgList->getSize())
		CmiAbort("CtgPup: global variable size changed during migration!\n");
	return g;
}

/** Install this set of globals. If g==NULL, returns to original globals. */
void CtgInstall(CtgGlobals g) {
	CtgGlobals *cur=&CpvAccess(_curCtg);
	CtgGlobals oldG=*cur;
	if (g==NULL) g=_ctgListGlobals;
	if (g == oldG) return;
        if (oldG) {
          _ctgList->read(oldG->data_seg);             /* store globals to own copy */
        }
	*cur=g;
	oldG->installed=0;
	_ctgList->install(g->data_seg);
	g->installed=1;
}

/** Delete this (not currently installed) set of globals. */
void CtgFree(CtgGlobals g) {
	if (g->installed) CmiAbort("CtgFree called on currently installed globals!\n");
	delete g;
}

CtgGlobals CtgCurrentGlobals(void){
	return CpvAccess(_curCtg);
}

void CtgInstall_var(CtgGlobals g, void *ptr) {
        CtgGlobals *cur=&CpvAccess(_curCtg);
	CtgGlobals oldG = *cur;
        if (oldG)
        _ctgList->read_var(oldG->data_seg, ptr);
        _ctgList->install_var(g->data_seg, ptr);             /* store globals to own copy */
}

void CtgUninstall_var(CtgGlobals g, void *ptr) {
        CtgGlobals *cur=&CpvAccess(_curCtg);
	CtgGlobals oldG = *cur;
        if (oldG)
        _ctgList->read_var(g->data_seg, ptr);             /* store globals to own copy */
        _ctgList->install_var(oldG->data_seg, ptr);             /* store globals to own copy */
}

#else     /* no ELF */

#include "global-nop.c"

#endif
