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
     VIRTUAL = 281,
     MIGRATABLE = 282,
     CREATEHERE = 283,
     CREATEHOME = 284,
     NOKEEP = 285,
     VOID = 286,
     CONST = 287,
     PACKED = 288,
     VARSIZE = 289,
     ENTRY = 290,
     FOR = 291,
     FORALL = 292,
     WHILE = 293,
     WHEN = 294,
     OVERLAP = 295,
     ATOMIC = 296,
     FORWARD = 297,
     IF = 298,
     ELSE = 299,
     CONNECT = 300,
     PUBLISHES = 301,
     IDENT = 302,
     NUMBER = 303,
     LITERAL = 304,
     CPROGRAM = 305,
     HASHIF = 306,
     HASHIFDEF = 307,
     INT = 308,
     LONG = 309,
     SHORT = 310,
     CHAR = 311,
     FLOAT = 312,
     DOUBLE = 313,
     UNSIGNED = 314
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
#define VIRTUAL 281
#define MIGRATABLE 282
#define CREATEHERE 283
#define CREATEHOME 284
#define NOKEEP 285
#define VOID 286
#define CONST 287
#define PACKED 288
#define VARSIZE 289
#define ENTRY 290
#define FOR 291
#define FORALL 292
#define WHILE 293
#define WHEN 294
#define OVERLAP 295
#define ATOMIC 296
#define FORWARD 297
#define IF 298
#define ELSE 299
#define CONNECT 300
#define PUBLISHES 301
#define IDENT 302
#define NUMBER 303
#define LITERAL 304
#define CPROGRAM 305
#define HASHIF 306
#define HASHIFDEF 307
#define INT 308
#define LONG 309
#define SHORT 310
#define CHAR 311
#define FLOAT 312
#define DOUBLE 313
#define UNSIGNED 314




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
#line 190 "y.tab.h"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;



