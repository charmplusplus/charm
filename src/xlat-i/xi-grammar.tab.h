/* A Bison parser, made by GNU Bison 2.1.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005 Free Software Foundation, Inc.

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

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

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
     CLASS = 272,
     INCLUDE = 273,
     STACKSIZE = 274,
     THREADED = 275,
     TEMPLATE = 276,
     SYNC = 277,
     IGET = 278,
     EXCLUSIVE = 279,
     IMMEDIATE = 280,
     SKIPSCHED = 281,
     INLINE = 282,
     VIRTUAL = 283,
     MIGRATABLE = 284,
     CREATEHERE = 285,
     CREATEHOME = 286,
     NOKEEP = 287,
     NOTRACE = 288,
     VOID = 289,
     CONST = 290,
     PACKED = 291,
     VARSIZE = 292,
     ENTRY = 293,
     FOR = 294,
     FORALL = 295,
     WHILE = 296,
     WHEN = 297,
     OVERLAP = 298,
     ATOMIC = 299,
     FORWARD = 300,
     IF = 301,
     ELSE = 302,
     CONNECT = 303,
     PUBLISHES = 304,
     PYTHON = 305,
     LOCAL = 306,
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
     UNSIGNED = 319
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
#define CLASS 272
#define INCLUDE 273
#define STACKSIZE 274
#define THREADED 275
#define TEMPLATE 276
#define SYNC 277
#define IGET 278
#define EXCLUSIVE 279
#define IMMEDIATE 280
#define SKIPSCHED 281
#define INLINE 282
#define VIRTUAL 283
#define MIGRATABLE 284
#define CREATEHERE 285
#define CREATEHOME 286
#define NOKEEP 287
#define NOTRACE 288
#define VOID 289
#define CONST 290
#define PACKED 291
#define VARSIZE 292
#define ENTRY 293
#define FOR 294
#define FORALL 295
#define WHILE 296
#define WHEN 297
#define OVERLAP 298
#define ATOMIC 299
#define FORWARD 300
#define IF 301
#define ELSE 302
#define CONNECT 303
#define PUBLISHES 304
#define PYTHON 305
#define LOCAL 306
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




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 17 "xi-grammar.y"
typedef union YYSTYPE {
  ModuleList *modlist;
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
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  PUPableClass *pupable;
  IncludeFile *includeFile;
  char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
} YYSTYPE;
/* Line 1447 of yacc.c.  */
#line 202 "y.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;



