
tree grammar MSA;

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

    public boolean accessorIsMSA(CharjAST ast)
    {
        return false;
    }

    public boolean expressionIsMSASync(CharjAST ast)
    {
        return false;
    }
}

topdown
    :   enterClass
    |   enterMethod
    |   possibleAccess
    |   possibleSync
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
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL
            | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) modifierList? type IDENT .*)
        {
            currentMethod = (MethodSymbol)$IDENT.def.type;
        }
    ;

exitMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL
            | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) modifierList? type IDENT .*)
        {
            currentMethod = null;
        }
    ;

possibleAccess
    :  ^(aa=ARRAY_ELEMENT_ACCESS .*)
        {
            if (accessorIsMSA($aa)) {
                currentMethod.hasMSA = true;
                currentMethod.addMSAAccess($aa);
            }
        }
    ;

possibleSync
    :   ^(mc=METHOD_CALL .*)
        {
            if (expressionIsMSASync($mc)) {
                currentMethod.hasMSA = true;
                currentMethod.addMSASync($mc);
            }
        }
    ;

modifierList
    :   ^(MODIFIER_LIST .*)
    ;

type
    :   VOID
    |   ^((OBJECT_TYPE|PROXY_TYPE|REFERENCE_TYPE|POINTER_TYPE
          |MESSAGE_TYPE|ARRAY_SECTION_TYPE) .*)
    ;
