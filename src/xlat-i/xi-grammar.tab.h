/* A Bison parser, made by GNU Bison 1.875.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002 Free Software Foundation, Inc.

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
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

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
     EXCLUSIVE = 278,
     IMMEDIATE = 279,
     SKIPSCHED = 280,
     INLINE = 281,
     VIRTUAL = 282,
     MIGRATABLE = 283,
     CREATEHERE = 284,
     CREATEHOME = 285,
     NOKEEP = 286,
     NOTRACE = 287,
     VOID = 288,
     CONST = 289,
     PACKED = 290,
     VARSIZE = 291,
     ENTRY = 292,
     FOR = 293,
     FORALL = 294,
     WHILE = 295,
     WHEN = 296,
     OVERLAP = 297,
     ATOMIC = 298,
     FORWARD = 299,
     IF = 300,
     ELSE = 301,
     CONNECT = 302,
     PUBLISHES = 303,
     PYTHON = 304,
     IDENT = 305,
     NUMBER = 306,
     LITERAL = 307,
     CPROGRAM = 308,
     HASHIF = 309,
     HASHIFDEF = 310,
     INT = 311,
     LONG = 312,
     SHORT = 313,
     CHAR = 314,
     FLOAT = 315,
     DOUBLE = 316,
     UNSIGNED = 317
   };
#endif
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
#define EXCLUSIVE 278
#define IMMEDIATE 279
#define SKIPSCHED 280
#define INLINE 281
#define VIRTUAL 282
#define MIGRATABLE 283
#define CREATEHERE 284
#define CREATEHOME 285
#define NOKEEP 286
#define NOTRACE 287
#define VOID 288
#define CONST 289
#define PACKED 290
#define VARSIZE 291
#define ENTRY 292
#define FOR 293
#define FORALL 294
#define WHILE 295
#define WHEN 296
#define OVERLAP 297
#define ATOMIC 298
#define FORWARD 299
#define IF 300
#define ELSE 301
#define CONNECT 302
#define PUBLISHES 303
#define PYTHON 304
#define IDENT 305
#define NUMBER 306
#define LITERAL 307
#define CPROGRAM 308
#define HASHIF 309
#define HASHIFDEF 310
#define INT 311
#define LONG 312
#define SHORT 313
#define CHAR 314
#define FLOAT 315
#define DOUBLE 316
#define UNSIGNED 317




#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 16 "xi-grammar.y"
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
/* Line 1249 of yacc.c.  */
#line 196 "y.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;



