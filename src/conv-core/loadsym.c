/* refun.c - replace libc functions in calling process

    Copyright (C) 2001 Victor Zandy <zandy@cs.wisc.edu>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This file exports one function:

      int replace_function(char *from, void *to);

          replace_function modifies the current process so that all
          subsequent calls to FROM, the name of a function in libc,
          are redirected to TO, a pointer to any function.

	  Returns 0 on success, -1 on failure.

    It (currently) only works on x86 Linux.

    Compile it with no special options to produce a .o that can be
    linked with a library or program, e.g. "cc -c refun.c".

    Send mail regarding this file to zandy@cs.wisc.edu.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#if CMK_HAS_ELF_H
#include <elf.h>

#include "converse.h"

#if CMK_AMD64 || CMK_CRAYXT
typedef Elf64_Addr    ELFXX_TYPE_Addr;
typedef Elf64_Dyn     ELFXX_TYPE_Dyn;
typedef Elf64_Rela    ELFXX_TYPE_Rel;
typedef Elf64_Sym     ELFXX_TYPE_Sym;
typedef Elf64_Shdr    ELFXX_TYPE_Shdr;
typedef Elf64_Ehdr    ELFXX_TYPE_Ehdr;
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
typedef Elf32_Shdr    ELFXX_TYPE_Shdr;
typedef Elf32_Ehdr    ELFXX_TYPE_Ehdr;
#define ELFXX_R_TYPE   ELF32_R_TYPE
#define ELFXX_R_SYM    ELF32_R_SYM
#define ELFXX_ST_TYPE  ELF32_ST_TYPE
#define CMK_DT_REL     DT_REL
#define CMK_DT_RELSZ   DT_RELSZ
#define is_elf_global(x)  ((x) == R_386_GLOB_DAT)
#endif

/* symbol table */
typedef struct symtab *symtab_t;
struct symlist {
	ELFXX_TYPE_Sym *sym;       /* symbols */
	char *str;            /* symbol strings */
	unsigned num;         /* number of symbols */
};
struct symtab {
	struct symlist *st;    /* "static" symbols */
	struct symlist *dyn;   /* dynamic symbols */
};


/* memory map for libraries */
#define MAX_NAME_LEN 256
#define MEMORY_ONLY  "[memory]"
struct mm {
	char name[MAX_NAME_LEN];
	unsigned long start, end;
};
static struct mm mm[50];
static int nmm;

/* FIXME: Why is this missing from unistd.h? */
extern ssize_t pread(int fd, void *buf, size_t count, off_t offset);

static struct symlist *
get_syms(int fd, ELFXX_TYPE_Shdr *symh, ELFXX_TYPE_Shdr *strh)
{
	struct symlist *sl, *ret;
	int rv;

	ret = NULL;
	sl = (struct symlist *) malloc(sizeof(struct symlist));
	if (!sl) {
		fprintf(stderr, "Out of memory\n");
		goto out;
	}
	sl->str = NULL;
	sl->sym = NULL;

	/* sanity */
	if (symh->sh_size % sizeof(ELFXX_TYPE_Sym)) { 
		fprintf(stderr, "elf_error\n");
		goto out;
	}

	/* symbol table */
	sl->num = symh->sh_size / sizeof(ELFXX_TYPE_Sym);
	sl->sym = (ELFXX_TYPE_Sym *) malloc(symh->sh_size);
	if (!sl->sym) {
		fprintf(stderr, "Out of memory\n");
		goto out;
	}
	rv = pread(fd, sl->sym, symh->sh_size, symh->sh_offset);
	if (0 > rv) {
		perror("read");
		goto out;
	}
	if (rv != symh->sh_size) {
		fprintf(stderr, "elf error\n");
		goto out;
	}

	/* string table */
	sl->str = (char *) malloc(strh->sh_size);
	if (!sl->str) {
		fprintf(stderr, "Out of memory\n");
		goto out;
	}
	rv = pread(fd, sl->str, strh->sh_size, strh->sh_offset);
	if (0 > rv) {
		perror("read");
		goto out;
	}
	if (rv != strh->sh_size) {
		fprintf(stderr, "elf error");
		goto out;
	}

	ret = sl;
out:
	return ret;
}

static int
do_load(int fd, symtab_t symtab)
{
	int rv;
	size_t size;
	ELFXX_TYPE_Ehdr ehdr;
	ELFXX_TYPE_Shdr *shdr = NULL, *p;
	ELFXX_TYPE_Shdr *dynsymh, *dynstrh;
	ELFXX_TYPE_Shdr *symh, *strh;
	char *shstrtab = NULL;
	int i;
	int ret = -1;
	
	/* elf header */
	rv = read(fd, &ehdr, sizeof(ehdr));
	if (0 > rv) {
		perror("read");
		goto out;
	}
	if (rv != sizeof(ehdr)) {
		fprintf(stderr, "elf error\n");
		goto out;
	}
	if (strncmp(ELFMAG, ehdr.e_ident, SELFMAG)) { /* sanity */
		fprintf(stderr, "not an elf\n");
		goto out;
	}
	if (sizeof(ELFXX_TYPE_Shdr) != ehdr.e_shentsize) { /* sanity */
		fprintf(stderr, "elf error\n");
		goto out;
	}

	/* section header table */
	size = ehdr.e_shentsize * ehdr.e_shnum;
	shdr = (ELFXX_TYPE_Shdr *) malloc(size);
	if (!shdr) {
		fprintf(stderr, "Out of memory\n");
		goto out;
	}
	rv = pread(fd, shdr, size, ehdr.e_shoff);
	if (0 > rv) {
		perror("read");
		goto out;
	}
	if (rv != size) {
		fprintf(stderr, "elf error");
		goto out;
	}
	
	/* section header string table */
	size = shdr[ehdr.e_shstrndx].sh_size;
	shstrtab = (char *) malloc(size);
	if (!shstrtab) {
		fprintf(stderr, "Out of memory\n");
		goto out;
	}
	rv = pread(fd, shstrtab, size, shdr[ehdr.e_shstrndx].sh_offset);
	if (0 > rv) {
		perror("read");
		goto out;
	}
	if (rv != size) {
		fprintf(stderr, "elf error\n");
		goto out;
	}

	/* symbol table headers */
	symh = dynsymh = NULL;
	strh = dynstrh = NULL;
	for (i = 0, p = shdr; i < ehdr.e_shnum; i++, p++)
		if (SHT_SYMTAB == p->sh_type) {
			if (symh) {
				fprintf(stderr, "too many symbol tables\n");
				goto out;
			}
			symh = p;
		} else if (SHT_DYNSYM == p->sh_type) {
			if (dynsymh) {
				fprintf(stderr, "too many symbol tables\n");
				goto out;
			}
			dynsymh = p;
		} else if (SHT_STRTAB == p->sh_type
			   && !strncmp(shstrtab+p->sh_name, ".strtab", 7)) {
			if (strh) {
				fprintf(stderr, "too many string tables\n");
				goto out;
			}
			strh = p;
		} else if (SHT_STRTAB == p->sh_type
			   && !strncmp(shstrtab+p->sh_name, ".dynstr", 7)) {
			if (dynstrh) {
				fprintf(stderr, "too many string tables\n");
				goto out;
			}
			dynstrh = p;
		}
	/* sanity checks */
	if ((!dynsymh && dynstrh) || (dynsymh && !dynstrh)) {
		fprintf(stderr, "bad dynamic symbol table");
		goto out;
	}
	if ((!symh && strh) || (symh && !strh)) {
		fprintf(stderr, "bad symbol table");
		goto out;
	}
	if (!dynsymh && !symh) {
		fprintf(stderr, "no symbol table");
		goto out;
	}

	/* symbol tables */
	if (dynsymh)
		symtab->dyn = get_syms(fd, dynsymh, dynstrh);
	if (symh)
		symtab->st = get_syms(fd, symh, strh);
	ret = 0;
out:
	free(shstrtab);
	free(shdr);
	return ret;
}

static symtab_t
load_symtab(char *filename)
{
	int fd;
	symtab_t symtab;

	symtab = (symtab_t) malloc(sizeof(*symtab));
	if (!symtab) {
		fprintf(stderr, "Out of memory\n");
		return NULL;
	}
	bzero(symtab, sizeof(*symtab));

	fd = open(filename, O_RDONLY);
	if (0 > fd) {
		perror("open");
		return NULL;
	}
	if (0 > do_load(fd, symtab)) {
		fprintf(stderr, "Error ELF parsing %s\n", filename);
		free(symtab);
		symtab = NULL;
	}
	close(fd);
	return symtab;
}

#define MYMAX(a,b)  (a>b?a:b)

static int
lookup2(struct symlist *sl, unsigned char type,
	char *name, unsigned long *val, unsigned int *size)
{
	ELFXX_TYPE_Sym *p;
	int len;
	int i;

	len = strlen(name);
	for (i = 0, p = sl->sym; i < sl->num; i++, p++) {
                int mlen = MYMAX(len, strlen(sl->str+p->st_name));       
		if (!strncmp(sl->str+p->st_name, name, mlen)
		    && ELFXX_ST_TYPE(p->st_info) == type) {
			*val = p->st_value;
                        *size = p->st_size;
			return 0;
		}
        }
	return -1;
}

static int
lookup_sym(symtab_t s, unsigned char type,
	   char *name, unsigned long *val, int *size)
{
	if (s->dyn && !lookup2(s->dyn, type, name, val, size))
		return 0;
	if (s->st && !lookup2(s->st, type, name, val, size))
		return 0;
	return -1;
}

static int
load_memmap()
{
	char raw[10000];
	char name[MAX_NAME_LEN];
	char *p;
	unsigned long start, end;
	struct mm *m;
	int fd, rv;
	int i;

	fd = open("/proc/self/maps", O_RDONLY);
	if (0 > fd) {
		fprintf(stderr, "Can't open /proc/self/maps for reading\n");
		return -1;
	}

	/* Zero to ensure data is null terminated */
	bzero(raw, sizeof(raw));
	rv = read(fd, raw, sizeof(raw));
	if (0 > rv) {
		perror("read");
		return -1;
	}
	if (rv >= sizeof(raw)) {
		fprintf(stderr, "Too many memory mapping\n");
		return -1;
	}
	close(fd);

	p = strtok(raw, "\n");
	m = mm;
	while (p) {
		/* parse current map line */
		rv = sscanf(p, "%08lx-%08lx %*s %*s %*s %*s %s\n",
			    &start, &end, name);
		p = strtok(NULL, "\n");

		if (rv == 2) {
			m = &mm[nmm++];
			m->start = start;
			m->end = end;
			strcpy(m->name, MEMORY_ONLY);
			continue;
		}

		/* search backward for other mapping with same name */
		for (i = nmm-1; i >= 0; i--) {
			m = &mm[i];
			if (!strcmp(m->name, name))
				break;
		}

		if (i >= 0) {
			if (start < m->start)
				m->start = start;
			if (end > m->end)
				m->end = end;
		} else {
			/* new entry */
			m = &mm[nmm++];
			m->start = start;
			m->end = end;
			strcpy(m->name, name);
		}
	}
	return 0;
}

/* Return non-zero iff NAME is the absolute pathname of the C library.
   This is a crude test and could stand to be sharpened. */
static int
match_libc(const char *name)
{
	char *p;

	p = strrchr(name, '/');
	if (!p)
		return 0;
	p++;
	if (strncmp("libc", p, 4))
		return 0;
	p += 4;

	/* here comes our crude test -> 'libc.so' or 'libc-[0-9]' */
	if (!strncmp(".so", p, 3) || (p[0] == '-' && isdigit(p[1])))
		return 1;
	return 0;
}

/* Find libc in calling process, storing no more than LEN-1 chars of
   its name in NAME and set START to its starting address.  If libc
   cannot be found return -1 and leave NAME and START untouched.
   Otherwise return 0 and null-terminated NAME. */
static int
find_my_libc(char *name, int len, unsigned long *start)
{
	int i;
	struct mm *m;

	if (!nmm && 0 > load_memmap()) {
		fprintf(stderr, "cannot read my memory map\n");
		return -1;
	}
	
	for (i = 0, m = mm; i < nmm; i++, m++) {
		if (!strcmp(m->name, MEMORY_ONLY))
			continue;
		if (match_libc(m->name))
			break; /* found it */
	}
	if (i >= nmm)
		/* not found */
		return -1;

	*start = m->start;
	strncpy(name, m->name, len);
	if (strlen(m->name) >= len)
		name[len-1] = '\0';
	return 0;
}

static int
patch(unsigned long from, unsigned long to)
{
	unsigned char *p;
	int *q;
	size_t pgsize;

	/* twiddle protection */
	pgsize = getpagesize();
	if (0 > mprotect((void *) (from & ~(pgsize - 1)), pgsize,
			 PROT_READ|PROT_WRITE|PROT_EXEC))
		return -1;

	/* opcode */
	p = (unsigned char *) from;
	*p++ = 0xe9;

	/* displacement */
	q = (int *) p;
	*q = to - (from + 5);

	/* FIXME: restore protection */
	return 0;
}

/* user-visible entry */

static symtab_t symtab;     /* libc symbol table */
static unsigned long libc;  /* libc start address */

int
lookup_obj_sym(char *name, unsigned long *val, int *size)
{
	if(-1==lookup_sym(symtab, STT_OBJECT, name, val, size) 
	 &&-1==lookup_sym(symtab, STT_NOTYPE, name, val, size))
		return -1;
	else 
		return 0;
}

int
init_symtab(char *exename)
{
	char libcname[128];

/*
	if (0 > find_my_libc(libcname, sizeof(libcname), &libc))
		return -1;
*/
	symtab = load_symtab(exename);
	/*symtab = load_symtab("cpi"); */
	if (!symtab)
		return -1;
	return 0;
}


#endif
