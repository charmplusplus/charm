/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

typedef union {
  ModuleList *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  EnType *rtype;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  Template *templat;
  TypeList *typelist;
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  char *strval;
  int intval;
} YYSTYPE;
#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	CHARE	261
#define	GROUP	262
#define	NODEGROUP	263
#define	ARRAY	264
#define	MESSAGE	265
#define	CLASS	266
#define	STACKSIZE	267
#define	THREADED	268
#define	TEMPLATE	269
#define	SYNC	270
#define	EXCLUSIVE	271
#define	VIRTUAL	272
#define	VOID	273
#define	PACKED	274
#define	VARSIZE	275
#define	ENTRY	276
#define	MAINCHARE	277
#define	IDENT	278
#define	NUMBER	279
#define	LITERAL	280
#define	INT	281
#define	LONG	282
#define	SHORT	283
#define	CHAR	284
#define	FLOAT	285
#define	DOUBLE	286
#define	UNSIGNED	287


extern YYSTYPE yylval;
