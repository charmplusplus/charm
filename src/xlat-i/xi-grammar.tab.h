typedef union {
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
  char *strval;
  int intval;
  Chare::attrib_t cattr;
} YYSTYPE;
#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	INITCALL	261
#define	CHARE	262
#define	MAINCHARE	263
#define	GROUP	264
#define	NODEGROUP	265
#define	ARRAY	266
#define	MESSAGE	267
#define	CLASS	268
#define	STACKSIZE	269
#define	THREADED	270
#define	TEMPLATE	271
#define	SYNC	272
#define	EXCLUSIVE	273
#define	VIRTUAL	274
#define	MIGRATABLE	275
#define	CREATEHERE	276
#define	CREATEHOME	277
#define	VOID	278
#define	CONST	279
#define	PACKED	280
#define	VARSIZE	281
#define	ENTRY	282
#define	IDENT	283
#define	NUMBER	284
#define	LITERAL	285
#define	CPROGRAM	286
#define	INT	287
#define	LONG	288
#define	SHORT	289
#define	CHAR	290
#define	FLOAT	291
#define	DOUBLE	292
#define	UNSIGNED	293


extern YYSTYPE yylval;
