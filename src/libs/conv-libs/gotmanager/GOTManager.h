
/*  Library creates a new global offset table for the program.
 *
 *  The global offset table is pointed to by the %ebx register.  To
 *  switch to the new GOT (global off set table) the data pointed to
 *  by ngot is moved to the register %ebx.
 *
 *  To compile on Linux IA-32
 *  gcc -fpic *.c 
 *
 *  Developed by Sameer Kumar (sameer@ks.uiuc.edu) 8/25/03
 *
 *  FIXME: Static strings do not work. A printf("hello world") does not
 *  work after the GOT switch because "hello world" is a
 *  static string. Similarly other static variables may not work. I
 *  will have to work more on this.
 *
 *  FIXME2: Sanity check. I am assuming that if a symbol is in the
 *  relocation table it is in the global offset table. A sanity check
 *  would be to get the address from the symbol table and look for it
 *  in the GOT. Pointers to remote function calls may be an exception
 *  to this.
 */

#include "converse.h"
#include <string.h>
#include <stdio.h>
#include <elf.h>
#include <stdlib.h>
#include <strings.h>
#include <errno.h>

#include "converse.h"
#include "pup.h"

#define ALIGN8(x)       (int)((~7)&((x)+7))

typedef Elf32_Addr ELF_TYPE_Addr;
typedef Elf32_Dyn  ELF_TYPE_Dyn;
typedef Elf32_Rel  ELF_TYPE_Rel;
typedef Elf32_Sym  ELF_TYPE_Sym;
 
extern ELF_TYPE_Addr _GLOBAL_OFFSET_TABLE_[]; //System global offset table
extern ELF_TYPE_Dyn _DYNAMIC[];      //The Dynamic section table pointer

#define DEBUG_GOT_MANAGER 1

class GOTManager{

    ELF_TYPE_Addr *ngot;     /* Pointer to the new global offset table*/
    int ngot_size;
    char *new_data_seg;   /* Pointer to the new global data segment*/
    int seg_size;

    //Creates a new global offset table
    void createNewGOT();

    int isValidSymbol(char *name);

 public:
    //Creates a new GOT and a data segment
    GOTManager();    

    //Swaps the GOT to the new GOT and returns the old got
    void *swapGOT();
    
    virtual void pup(PUP::er &p);

    static void restoreGOT(void *oldgot);        
};
