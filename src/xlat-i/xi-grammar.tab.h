/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     MODULE = 258,
     MAINMODULE = 259,
     EXTERN = 260,
     READONLY = 261,
     INITCALL = 262,
     INITNODE = 263,
     INITPROC = 264,
     PUPABLE = 265,
     CHARE = 266,
     MAINCHARE = 267,
     GROUP = 268,
     NODEGROUP = 269,
     ARRAY = 270,
     MESSAGE = 271,
     CONDITIONAL = 272,
     CLASS = 273,
     INCLUDE = 274,
     STACKSIZE = 275,
     THREADED = 276,
     TEMPLATE = 277,
     SYNC = 278,
     IGET = 279,
     EXCLUSIVE = 280,
     IMMEDIATE = 281,
     SKIPSCHED = 282,
     INLINE = 283,
     VIRTUAL = 284,
     MIGRATABLE = 285,
     CREATEHERE = 286,
     CREATEHOME = 287,
     NOKEEP = 288,
     NOTRACE = 289,
     VOID = 290,
     CONST = 291,
     PACKED = 292,
     VARSIZE = 293,
     ENTRY = 294,
     FOR = 295,
     FORALL = 296,
     WHILE = 297,
     WHEN = 298,
     OVERLAP = 299,
     ATOMIC = 300,
     IF = 301,
     ELSE = 302,
     PYTHON = 303,
     LOCAL = 304,
     NAMESPACE = 305,
     USING = 306,
     IDENT = 307,
     NUMBER = 308,
     LITERAL = 309,
     CPROGRAM = 310,
     HASHIF = 311,
     HASHIFDEF = 312,
     INT = 313,
     LONG = 314,
     SHORT = 315,
     CHAR = 316,
     FLOAT = 317,
     DOUBLE = 318,
     UNSIGNED = 319,
     ACCEL = 320,
     READWRITE = 321,
     WRITEONLY = 322,
     ACCELBLOCK = 323,
     MEMCRITICAL = 324,
     REDUCTIONTARGET = 325,
     CASE = 326
   };
#endif
/* Tokens.  */
#define MODULE 258
#define MAINMODULE 259
#define EXTERN 260
#define READONLY 261
#define INITCALL 262
#define INITNODE 263
#define INITPROC 264
#define PUPABLE 265
#define CHARE 266
#define MAINCHARE 267
#define GROUP 268
#define NODEGROUP 269
#define ARRAY 270
#define MESSAGE 271
#define CONDITIONAL 272
#define CLASS 273
#define INCLUDE 274
#define STACKSIZE 275
#define THREADED 276
#define TEMPLATE 277
#define SYNC 278
#define IGET 279
#define EXCLUSIVE 280
#define IMMEDIATE 281
#define SKIPSCHED 282
#define INLINE 283
#define VIRTUAL 284
#define MIGRATABLE 285
#define CREATEHERE 286
#define CREATEHOME 287
#define NOKEEP 288
#define NOTRACE 289
#define VOID 290
#define CONST 291
#define PACKED 292
#define VARSIZE 293
#define ENTRY 294
#define FOR 295
#define FORALL 296
#define WHILE 297
#define WHEN 298
#define OVERLAP 299
#define ATOMIC 300
#define IF 301
#define ELSE 302
#define PYTHON 303
#define LOCAL 304
#define NAMESPACE 305
#define USING 306
#define IDENT 307
#define NUMBER 308
#define LITERAL 309
#define CPROGRAM 310
#define HASHIF 311
#define HASHIFDEF 312
#define INT 313
#define LONG 314
#define SHORT 315
#define CHAR 316
#define FLOAT 317
#define DOUBLE 318
#define UNSIGNED 319
#define ACCEL 320
#define READWRITE 321
#define WRITEONLY 322
#define ACCELBLOCK 323
#define MEMCRITICAL 324
#define REDUCTIONTARGET 325
#define CASE 326




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 23 "xi-grammar.y"
{
  AstChildren<Module> *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  EntryList *entrylist;
  Parameter *pname;
  ParamList *plist;
  Template *templat;
  TypeList *typelist;
  AstChildren<Member> *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  PUPableClass *pupable;
  IncludeFile *includeFile;
  const char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
  WhenConstruct *when;
  XStr* xstrptr;
  AccelBlock* accelBlock;
}
/* Line 1529 of yacc.c.  */
#line 230 "y.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

