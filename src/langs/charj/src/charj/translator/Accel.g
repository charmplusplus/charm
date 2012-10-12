
tree grammar Accel;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
}

@header {
    package charj.translator;
}

@members {
    Scope currentScope;
    ClassSymbol currentClass = null;
    MethodSymbol currentMethod = null;
}

topdown
    :   enterClass
    |   enterMethod
    |   atoms
    ;

bottomup
    :   exitMethod
    |   exitClass
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

enterClass
    :   ^(TYPE classType IDENT
        (^('extends' .*))?
        (^('implements' .*))?
        (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION | DIVCON_METHOD_DECL |
            OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = (ClassSymbol)$IDENT.def.type;
            currentClass.isMainChare = $classType.text.equals("mainchare");
        }
    ;

exitClass
    :   ^(TYPE classType IDENT
        (^('extends' .*))?
        (^('implements' .*))?
        (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION | DIVCON_METHOD_DECL |
            OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = null;
        }
    ;

enterMethod
    :   ^(ENTRY_FUNCTION_DECL ^(ml=MODIFIER_LIST .*) type IDENT .*)
        {
            currentMethod = (MethodSymbol)$IDENT.def.type;
            if ($ml.hasAccelModifier()) currentMethod.accel = true;
        }
    ;

exitMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL
            | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*)
        {
            currentMethod = null;
        }
    ;

type
    :   VOID
    |   ^((OBJECT_TYPE|PROXY_TYPE|REFERENCE_TYPE|POINTER_TYPE
          |MESSAGE_TYPE|ARRAY_SECTION_TYPE) .*)
    ;

atoms
    :  (id=IDENT) {
            if (currentMethod != null && currentMethod.accel) {
                currentMethod.addAccelIdent($id);
            }
       }
    ;
