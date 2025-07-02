/*  Library to manage the program's global offset table.
 *
 *  The global offset table (GOT) is a static, linker-generated 
 *  table used to look up the locations of dynamic stuff
 *  like subroutines or global data.
 *
 *  It's only generated and used if you're on an ELF binary
 *  format machine, and compile with "-fpic", for "position 
 *  independent code"; otherwise the code includes direct 
 *  references to subroutines and data.
 *
 *  During execution of a single routine, the GOT is pointed to
 *  by %ebx.  Changing %ebx only affects global access within
 *  the current subroutine, so we have to change the single, 
 *  static copy of the GOT to switch between sets of global variables.
 *  This is slow, but because the GOT just contains pointers,
 *  it's not as slow as copying the data in and out of the GOT.
 *
 
The ELF GOT layout is described in excruciating detail
by the Sun documentation, at 
   Solaris 2.5 Software Developer AnswerBook >> 
      Linker and Libraries Guide >> 
        6 Object Files >> 
	  File Format

A more readable summary is at:  
  http://www.iecc.com/linker/linker08.html

 *
 *  Developed by Sameer Kumar (sameer@ks.uiuc.edu) 8/25/03
 *  Made functional by Orion Lawlor (olawlor@acm.org) 2003/9/16
 *  Made working on AMD64 by Gengbin Zheng 12/20/2005
 *
 *  FIXME2: Sanity check. I am assuming that if a symbol is in the
 *  relocation table it is in the global offset table. A sanity check
 *  would be to get the address from the symbol table and look for it
 *  in the GOT. Pointers to remote function calls may be an exception
 *  to this.
 */

#include "converse.h"

/* conv-config.h, included in converse.h, defines CMI_SWAPGLOBALS.
 * -swapglobals only works on ELF systems and in non-SMP mode. */
#if CMI_SWAPGLOBALS

#include "cklists.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#if CMK_HAS_REGEX_H
#include <regex.h>
#endif
#include <vector>
#include <algorithm>
#include "converse.h"
#include "pup.h"

#include <elf.h>

#define DEBUG_GOT_MANAGER 0

#define UNPROTECT_GOT     1

#if UNPROTECT_GOT && CMK_HAS_MPROTECT
#include <sys/mman.h>
#if CMK_HAS_GETPAGESIZE
#include <unistd.h>
#endif
#endif

#ifdef __INTEL_COMPILER
#define ALIGN_GOT(x)       (long)((~15)&((x)+15))
#else
#define ALIGN_GOT(x)       ALIGN8(x)
#endif

CpvDeclare(int, CmiPICMethod);

#if CMK_64BIT
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


/**
	Method to read blacklist of variables that should not be 
	identified as global variables
	**/

std::vector<char *>  _blacklist;
static bool loaded = false;
extern int quietModeRequested;

static void readBlacklist()
{
  if (loaded) return;
  const char *fname = "blacklist";
  FILE *bl = fopen(fname, "r");
  if (bl == NULL){
		if (!quietModeRequested && CmiMyPe() == 0) {
			CmiPrintf("WARNING: Running swapglobals without blacklist, globals from libraries might be unnecessarily swapped.\n");
		}
		loaded = true;
		return;
  }
  printf("Loading blacklist from file \"%s\" ... \n", fname);
  while (!feof(bl)){
    char name[512];
    if (fscanf(bl, "%511s\n", name) != 1) {
      CmiAbort("Swapglobals> reading blacklist file failed!");
    }
     _blacklist.push_back(strdup(name));
  }
  fclose(bl);
  loaded = true;
}




/****************** Global Variable Understanding *********************/
/**
 Keeps a list of global variables.
*/
class CtgGlobalList
{
  size_t datalen; ///< Number of bytes in the table of global data.
  struct CtgRec {
    ELFXX_TYPE_Addr *got; ///< Points to our entry in the GOT.
    size_t off; ///< Our byte offset into the table of global data.
    CtgRec() {got=NULL;}
    CtgRec(ELFXX_TYPE_Addr *got_,int off_) :got(got_), off(off_) {}
  };
  std::vector<CtgRec> rec;
public:
  /**
   Analyze the current set of global variables, determine 
   which are user globals and which are system globals, 
   and store the list of user globals. 
  */
  CtgGlobalList();
  
  /// Return the number of bytes needed to store our global data.
  inline size_t getSize(void) const {return datalen;}
  
  /// Copy the current set of global data into this set,
  ///   which must be getSize() bytes.
  void read(void *datav) const;
  
  /// Point at this set of global data (must be getSize() bytes).
  inline void install(void *datav) const {
    intptr_t data = (intptr_t)datav;
    int nRec = rec.size();
    for (int i=0;i<nRec;i++)
      *(rec[i].got)=(ELFXX_TYPE_Addr)(data+rec[i].off);
  }
  
private:
  /* Return 1 if this is the name of a user global variable;
     0 if this is the name of some other (Charm or system)
     global variable. */
  int isUserSymbol(const char *name);
};

int match(const char *string, const char *pattern) {
#if CMK_HAS_REGEX_H
  int status;

  regex_t re;
  if (regcomp(&re, pattern, REG_EXTENDED|REG_NOSUB) != 0) {
    CmiAbort("error compiling regex");
  }
  status = regexec(&re, string, (size_t) 0, NULL, 0);
  regfree(&re);
  if (status == 0) {
    return 1;
  } else if (status == REG_NOMATCH) {
    return 0;
  }
  perror("error in match\n");
  return 0;
#else
  CmiPrinf("Warning: elfgot.C::match() is not implemented!\n");
  return 0;
#endif
}

int CtgGlobalList::isUserSymbol(const char *name) {
    // return 1;
    if((strncmp("_", name, 1) == 0) || (strncmp("Cpv_", name, 4) == 0)
       || (strncmp("Csv_", name, 4) == 0) || (strncmp("Ctv_", name, 4) == 0)
       || (strncmp("Bnv_", name, 4) == 0) || (strncmp("Bpv_", name, 4) == 0)
       || (strncmp("ckout", name, 5) == 0) || (strncmp("stdout", name, 6) == 0)
       || (strncmp("ckerr", name, 5) == 0)
       || (strncmp("environ", name, 7) == 0)
       || (strncmp("pthread", name, 7) == 0)
       || (strncmp("stderr", name, 6) == 0) || (strncmp("stdin", name, 5) == 0)) {
#ifdef CMK_GFORTRAN
        if (match(name, "__.*_MOD_.*")) return 1;
#endif
        return 0;
    }
    
	/**
		if the name is on the blacklist, it is not a user symbol
	*/
	for(unsigned int i=0;i<_blacklist.size();i++){
		if(strlen(name) == strlen(_blacklist[i]) && strncmp(name,_blacklist[i],strlen(name)) == 0){
			return 0;
		}
	}
		
    return 1;
}

void CtgGlobalList::read(void *datav) const {
    char *data=(char *)datav;
    int nRec = rec.size();
    size_t size;
    for (int i=0;i<nRec-1;i++) { // First nRec-1 elements:
      size = rec[i+1].off-rec[i].off;
      memcpy(data+rec[i].off, (void *)*rec[i].got, size);
    }
    // Last element:
    if (nRec > 0) {
      size = datalen-rec[nRec-1].off;
      memcpy(data+rec[nRec-1].off, (void *)(uintptr_t)*rec[nRec-1].got, size);
    }
}

/**
   Analyze the current set of global variables, determine 
   which are user globals and which are system globals, 
   and store the list of user globals.
 */
CtgGlobalList::CtgGlobalList() {
    datalen=0;
    
    int count;
    size_t relt_size = 0;
    int type, symindx;
    char *sym_name;
    ELFXX_TYPE_Rel *relt=NULL;       //Relocation table
    ELFXX_TYPE_Sym *symt=NULL;       //symbol table
    char *str_tab=NULL;         //String table

    // Find tables and sizes of tables from the dynamic segment table
    for(count = 0; _DYNAMIC[count].d_tag != 0; ++count) {
	switch(_DYNAMIC[count].d_tag) {
	    case CMK_DT_REL:
		relt = (ELFXX_TYPE_Rel *) _DYNAMIC[count].d_un.d_ptr;
		break;
	    case CMK_DT_RELSZ:
		relt_size = _DYNAMIC[count].d_un.d_val/ sizeof(ELFXX_TYPE_Rel);
		break;
	    case DT_SYMTAB:
		symt = (ELFXX_TYPE_Sym *) _DYNAMIC[count].d_un.d_ptr;
		break;
	    case DT_STRTAB:
		str_tab = (char *)_DYNAMIC[count].d_un.d_ptr;
		break;
	}
    }

    int padding = 0;

#if UNPROTECT_GOT && CMK_HAS_MPROTECT
    const size_t pagesize = CmiGetPageSize();
#endif

    // Figure out which relocation data entries refer to global data:
    for(count = 0; count < relt_size; count ++) {
        type = ELFXX_R_TYPE(relt[count].r_info);
        symindx = ELFXX_R_SYM(relt[count].r_info);
 
        if(!is_elf_global(type))
	    continue; /* It's not global data */

	sym_name = str_tab + symt[symindx].st_name;

#if DEBUG_GOT_MANAGER
	printf("relt[%d]= %s: %d bytes, %p sym, R_==%d\n", count, sym_name, 
	       symt[symindx].st_size, (void *)symt[symindx].st_value, type);
#endif

	if(ELFXX_ST_TYPE(symt[symindx].st_info) != STT_OBJECT &&
	   ELFXX_ST_TYPE(symt[symindx].st_info) != STT_NOTYPE
#if 0
#ifdef __INTEL_COMPILER
          && ELFXX_ST_TYPE(symt[symindx].st_info) != STT_FUNC
#endif
#endif
                 ) /* ? */
	    continue;

	if(strcmp(sym_name, "_DYNAMIC") == 0 ||
	   strcmp(sym_name, "__gmon_start__") == 0 ||
	   strcmp(sym_name, "_GLOBAL_OFFSET_TABLE_") == 0)
	    continue; /* It's system data */

	if (!isUserSymbol(sym_name))
	    continue;

	// It's got the right name-- it's a user global
	size_t size = symt[symindx].st_size;
	size_t gSize = ALIGN_GOT(size);
	padding += gSize - size;
	ELFXX_TYPE_Addr *gGot=(ELFXX_TYPE_Addr *)(uintptr_t)relt[count].r_offset;

#if DEBUG_GOT_MANAGER
	printf("   -> %s is a user global, of size %d, at %p\n",
	       sym_name, size, (void *)*gGot);
#endif
	if ((void *)(uintptr_t)*gGot != (void *)(uintptr_t)symt[symindx].st_value)
	    CmiAbort("CtgGlobalList: symbol table and GOT address mismatch!\n");

#if UNPROTECT_GOT && CMK_HAS_MPROTECT
	static void *last = NULL;
        void *pg = (void*)(uintptr_t)(((size_t)gGot) & ~(pagesize-1));
        if (pg != last) {
            mprotect(pg, pagesize, PROT_READ | PROT_WRITE);
            last = pg;
        }
#endif

	rec.emplace_back(gGot, datalen);
	datalen+=gSize;
    }

#if DEBUG_GOT_MANAGER   
    printf("relt has %d entries, %d of which are user globals\n\n", 
	   relt_size, rec.size());
    printf("Globals take %d bytes (padding bytes: %d)\n", datalen, padding);
#endif
}

/****************** Global Variable Storage and Swapping *********************/
CpvStaticDeclare(CtgGlobals,_curCtg);

/// Singleton object describing our global variables:
static CtgGlobalList *_ctgList=NULL;
/// Singleton object describing the original values for the globals.
static CtgGlobalStruct _ctgListGlobals;

/** Initialize the globals support (called on each processor). */
void CtgInit(void) {
	CpvInitialize(int, CmiPICMethod);
	CpvAccess(CmiPICMethod) = CMI_PIC_ELFGOT;
	CpvInitialize(CtgGlobals,_curCtg);
	
	if (!_ctgList) 
	{
	/*
	  First call on this node: parse out our globals:
	*/
		readBlacklist();
		CtgGlobalList *l=new CtgGlobalList;
		CtgGlobalStruct * g = &_ctgListGlobals;
		if (!quietModeRequested && CmiMyPe() == 0) {
			CmiPrintf("Charm++> -swapglobals enabled for automatic privatization of global variables.\n"
				"WARNING> -swapglobals does not handle static variables.\n");
		}
		
		g->data_seg = malloc(l->getSize());
		l->read(g->data_seg);
		l->install(g->data_seg);
		_ctgList=l;
	}

	CpvAccess(_curCtg)=_ctgListGlobals;

#if CMK_ERROR_CHECKING
  // Could ensure _ctgList->getSize() is the same on all logical nodes.
#endif
}

size_t CtgGetSize()
{
  return _ctgList->getSize();
}

/** Copy the current globals into this new set */
CtgGlobals CtgCreate(void * buffer) {
	CtgGlobalStruct g = { buffer };
	_ctgList->read(buffer);
	return g;
}

/** Install this set of globals. */
void CtgInstall(CtgGlobals g) {
	CtgGlobals * cur = &CpvAccess(_curCtg);
	if (cur->data_seg == g.data_seg)
	  return;
	cur->data_seg = g.data_seg;
	_ctgList->install(g.data_seg);
}

void CtgUninstall() {
	CtgInstall(_ctgListGlobals);
}

CtgGlobals CtgCurrentGlobals(void){
	return CpvAccess(_curCtg);
}

#else // CMI_SWAPGLOBALS

#include "global-nop.C"

#endif
