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
  MsgVar *mv;
  MsgVarList *mvlist;
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
#define	MIGRATABLE	269
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
