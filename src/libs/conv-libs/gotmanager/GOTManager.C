#include "GOTManager.h"

GOTManager::GOTManager() {
    createNewGOT();
}

void GOTManager::createNewGOT(){
    int count;
    int off, pos;
    int relt_size = 0, pltrelt_size = 0;
    int global_relt_size = 0, global_pltrelt_size = 0;
        
    int type, symindx, total_size;
    char *sym_name;
    int dynamic_pos;
    int got_pos;
    int gmon_pos;

    seg_size = 0;

    ELF_TYPE_Rel *relt;       //Relocation table
    ELF_TYPE_Rel *pltrelt;    //Procedure relocation table
    ELF_TYPE_Sym *symt;       //symbol table
    char *str_tab;         //String table

    //strcpy(hello, "hello world %d, %d, %d, %u\n");

    /*Point tables and sizes of tables to the correct values
      from the dynamic segment table*/
    count = 0;
    while(_DYNAMIC[count].d_tag != 0){

        if(_DYNAMIC[count].d_tag == DT_REL)
            relt = (ELF_TYPE_Rel *) _DYNAMIC[count].d_un.d_ptr;

        else if(_DYNAMIC[count].d_tag == DT_RELSZ)
            relt_size = _DYNAMIC[count].d_un.d_val/ sizeof(ELF_TYPE_Rel);

        else if(_DYNAMIC[count].d_tag == DT_SYMTAB)
            symt = (ELF_TYPE_Sym *) _DYNAMIC[count].d_un.d_ptr;

        else if(_DYNAMIC[count].d_tag == DT_JMPREL)
            pltrelt = (ELF_TYPE_Rel *) _DYNAMIC[count].d_un.d_ptr;

        else if(_DYNAMIC[count].d_tag == DT_PLTRELSZ)
            pltrelt_size = _DYNAMIC[count].d_un.d_val/ sizeof(ELF_TYPE_Rel);

        else if(_DYNAMIC[count].d_tag == DT_STRTAB)
            str_tab = (char *)_DYNAMIC[count].d_un.d_ptr;

        count ++;
    }

    /*Compute the Number of global relocation data entries*/
    for(count = 0; count < relt_size; count ++){
        type = ELF32_R_TYPE(relt[count].r_info);
        symindx = ELF32_R_SYM(relt[count].r_info);
        
        if(type == R_386_GLOB_DAT){
            sym_name = str_tab + symt[symindx].st_name;
            if(strcmp(sym_name, "_DYNAMIC") == 0)
                dynamic_pos = count;
            else if(strcmp(sym_name, "__gmon_start__") == 0)
                gmon_pos = count;
            else if(strcmp(sym_name, "_GLOBAL_OFFSET_TABLE_") == 0)
                got_pos = count;
            else {
                //Look for data and not function symbols
                if(ELF32_ST_TYPE(symt[symindx].st_info) == STT_OBJECT &&
                   isValidSymbol(sym_name) ) {
#if DEBUG_GOT_MANAGER   
                    sym_name = str_tab + symt[symindx].st_name;
                    printf("%d %s %d %d\n", symindx, sym_name, 
                           symt[symindx].st_size, symt[symindx].st_value);
#endif
                    seg_size += ALIGN8(symt[symindx].st_size);
                }
                    
                global_relt_size ++;
            }
        }
    }
    
    /*Compute the Number of global relocation procedure entries*/
    for(count = 0; count < pltrelt_size; count ++){
        type = ELF32_R_TYPE(pltrelt[count].r_info);
        symindx = ELF32_R_SYM(pltrelt[count].r_info);

        //printf("%d %d\n", symindx, type);

        if(type == R_386_JMP_SLOT){
#if DEBUG_GOT_MANAGER   
            //sym_name = str_tab + symt[symindx].st_name;
            //printf("%d %s %d %d\n", symindx, sym_name, 
            //     symt[symindx].st_size, symt[symindx].st_value);
#endif
            global_pltrelt_size ++;
        }
    }

#if DEBUG_GOT_MANAGER   
    printf("\n\n");
    printf("relt_size = %d, pltrelt_size = %d\n", global_relt_size,
           global_pltrelt_size);
    printf("Creating data segment of size %d\n", seg_size);
#endif

    /*Compute the size of the got and allocate memory for the new got*/
    total_size = 3 + relt_size + pltrelt_size;
    ngot = (ELF_TYPE_Addr *) calloc(total_size * sizeof(ELF_TYPE_Addr), 1);
    total_size = 3 + global_relt_size + global_pltrelt_size;
    //Copying the global offset table to the new global offset table
    for(count = 0; count < total_size; count ++)
        ngot[count] =  _GLOBAL_OFFSET_TABLE_[count];

    ngot_size = total_size;

    //Allocating memory for the new data segment
    new_data_seg = (char *) malloc(seg_size);
    pos = 3 + global_pltrelt_size;
    off = 0;

    //Copy Global offset table and global data
    for(count = 0; count < global_relt_size; count ++){
        type = ELF32_R_TYPE(relt[count].r_info);
        symindx = ELF32_R_SYM(relt[count].r_info);

        if(type == R_386_GLOB_DAT){
            if(count == dynamic_pos) //Dont change pointer to _DYNAMIC
                pos ++;
            else if(count == got_pos){
                ngot[pos ++] = (ELF_TYPE_Addr)ngot;
            }
            //A strange weirdness,
            //this variable is not present in the GOT
            else if(count == gmon_pos);
            
            else {
                if(ELF32_ST_TYPE(symt[symindx].st_info) == STT_OBJECT
                   && isValidSymbol(sym_name) ) {
                    ngot[pos ++] = (ELF_TYPE_Addr)
                        ((char *)new_data_seg + off);
                    off += ALIGN8(symt[symindx].st_size);
                    
                    memcpy((void *)ngot[pos-1], 
                           (void *)_GLOBAL_OFFSET_TABLE_[pos-1],
                           symt[symindx].st_size);
                }
                else pos ++; //Leave all function pointers alone
                             //and Charm++ system globals alone
            }
        }
    }

#if DEBUG_GOT_MANAGER
    /*
      for(count = 0; count < 3 + global_relt_size + global_pltrelt_size; 
      count ++)
      printf("[%d] GOT = %u  NGOT = %u\n", count, 
      _GLOBAL_OFFSET_TABLE_[count], ngot[count]);
    */
#endif
}

int GOTManager::isValidSymbol(char *name){
    
    if((strncmp("_", name, 2) == 0) || (strncmp("Cpv_", name, 4) == 0)
       || (strncmp("Csv_", name, 4) == 0) || (strncmp("Ctv_", name, 4) == 0)
       || (strncmp("ckout", name, 5) == 0) || (strncmp("stdout", name, 6) == 0)
       || (strncmp("stderr", name, 6) == 0))
        return 0;
    
    return 1;
}

ELF_TYPE_Addr *_gotmanager_temp;

void * GOTManager::swapGOT(){
    ELF_TYPE_Addr *oldgot= 0, *new_got = 0;
    
    oldgot = _GLOBAL_OFFSET_TABLE_;    

    _gotmanager_temp = ngot;
    asm("movl _gotmanager_temp, %ebx;");        
    
    //printf("testing values %d, %d\n", oldgot, new_got);

    return (void *)oldgot;
}

void GOTManager::restoreGOT(void *oldgot){

    _gotmanager_temp = (ELF_TYPE_Addr *) oldgot;
    asm("movl _gotmanager_temp, %ebx;");      
}


void GOTManager::pup(PUP::er &p) {
    p | ngot_size;
    p | seg_size;

    p(new_data_seg, seg_size);
    p(ngot, ngot_size);
}

