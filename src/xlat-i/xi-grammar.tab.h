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
#define	MODULE	258
#define	MAINMODULE	259
#define	EXTERN	260
#define	READONLY	261
#define	CHARE	262
#define	GROUP	263
#define	NODEGROUP	264
#define	ARRAY	265
#define	MESSAGE	266
#define	CLASS	267
#define	STACKSIZE	268
#define	THREADED	269
#define	TEMPLATE	270
#define	SYNC	271
#define	EXCLUSIVE	272
#define	VIRTUAL	273
#define	VOID	274
#define	PACKED	275
#define	VARSIZE	276
#define	ENTRY	277
#define	MAINCHARE	278
#define	IDENT	279
#define	NUMBER	280
#define	LITERAL	281
#define	INT	282
#define	LONG	283
#define	SHORT	284
#define	CHAR	285
#define	FLOAT	286
#define	DOUBLE	287
#define	UNSIGNED	288


extern YYSTYPE yylval;
