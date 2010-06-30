
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
    boolean inBlock = false;
}

topdown
    :   enterClass
    |   enterBlock
    |   enterDefaultConstructor
    |   enterMigrationConstructor
    |   varDeclaration
    ;

bottomup
    :   exitBlock
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
    :    ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL)
            (^(MODIFIER_LIST .*))? .
            FORMAL_PARAM_LIST ^(BLOCK .*))
        {
            if (currentClass != null) {
                currentClass.hasDefaultConstructor = true;
            }
        }
    ;

enterMigrationConstructor
    :    ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL)
            (^(MODIFIER_LIST .*))? .
                ^(FORMAL_PARAM_LIST ^(FORMAL_PARAM_STD_DECL
                    ^(POINTER_TYPE ^(QUALIFIED_TYPE_IDENT IDENT)) .
                )) ^(BLOCK .*))
        {
            if (currentClass != null && $IDENT.text.equals("CkMigrateMessage")) {
                currentClass.hasMigrationConstructor = true;
            }
        }
    ;

enterBlock 
    :   ^(BLOCK .*)
        {
            inBlock = true;
        }
    ;

exitBlock 
    :   ^(BLOCK .*)
        {
            inBlock = false;
        }
    ;

varDeclaration
    :   ^(VAR_DECLARATOR ^(IDENT .*) (expr=.)? )
        {

            if (!inBlock && currentClass != null && $expr != null) {
                System.out.println("FOUND EXPR");
                currentClass.initializers.add(new VariableInitializer($expr, $IDENT));
            }

            if (!inBlock && currentClass != null) {
                currentClass.varsToPup.add($IDENT);
            }
        }
    ;


