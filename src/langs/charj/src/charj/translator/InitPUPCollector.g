
tree grammar InitPUPCollector;

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
    boolean inMethod = false;
}

topdown
    :   enterClass
    |   enterMethod
    |   enterDefaultConstructor
    |   enterMigrationConstructor
    |   varDeclaration
    ;

bottomup
    :   exitMethod
    ;

enterClass
    :   ^(TYPE .* IDENT
            (^('extends' .*))?
            (^('implements' .*))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = (ClassSymbol)$IDENT.def.type;
        }
    ;

enterDefaultConstructor
    :    FORMAL_PARAM_LIST
        {
            if (($FORMAL_PARAM_LIST.hasParentOfType(CONSTRUCTOR_DECL) ||
                 $FORMAL_PARAM_LIST.hasParentOfType(ENTRY_CONSTRUCTOR_DECL)) &&
                currentClass != null) {
                currentClass.hasDefaultConstructor = true;
            }
        }
    ;

enterMigrationConstructor
    :    ^(FORMAL_PARAM_LIST ^(FORMAL_PARAM_STD_DECL
                ^(POINTER_TYPE ^(QUALIFIED_TYPE_IDENT IDENT)) .
            ))
        {
            if (($FORMAL_PARAM_LIST.hasParentOfType(CONSTRUCTOR_DECL) ||
                 $FORMAL_PARAM_LIST.hasParentOfType(ENTRY_CONSTRUCTOR_DECL)) &&
                currentClass != null && $IDENT.text.equals("CkMigrateMessage")) {
                currentClass.hasMigrationConstructor = true;
            }
        }
    ;

enterMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL
            | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*)
        {
            inMethod = true;
        }
    ;

exitMethod
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL
            | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*)
        {
            inMethod = false;
        }
    ;

varDeclaration
    :   ^(VAR_DECLARATOR ^(IDENT .*) (expr=.)? )
        {

            if (!inMethod && currentClass != null && $expr != null) {
                System.out.println("FOUND EXPR");
                currentClass.initializers.add(new VariableInitializer($expr, $IDENT));
            }

            if (!inMethod && currentClass != null) {
                currentClass.varsToPup.add($IDENT);
            }
        }
    ;


