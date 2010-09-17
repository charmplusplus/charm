
tree grammar AtomicBlocks;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
}

@header {
package charj.translator;
import java.util.Iterator;
}

@members {
    SymbolTable symtab;
    Scope currentScope;
    ClassSymbol currentClass = null;

    public SymbolResolver(TreeNodeStream input, SymbolTable symtab) {
        this(input);
        this.symtab = symtab;
        this.currentScope = symtab.getDefaultPkg();
    }
}


topdown
    :   enterClass
    |   enterMethod
    |   varDeclaration
    |   expression
    |   assignment
    ;

bottomup
    :   exitClass
    ;

enterMethod
@init {
boolean entry = false;
}
    :   ^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            type IDENT .*)
        {
            $IDENT.def.type = $type.sym;
            $IDENT.symbolType = $IDENT.def.type;
        }
    |   ^((CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL {entry = true;})
            (^(MODIFIER_LIST .*))?
            (^(GENERIC_TYPE_PARAM_LIST .*))? 
            IDENT .*)
        {
            $IDENT.def.type = (ClassSymbol)$IDENT.def.scope;
            $IDENT.symbolType = $IDENT.def.type;
        }
    ;

enterClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        {
            currentClass = (ClassSymbol)$IDENT.def.type;
        }
    ;

exitClass
    :   ^(TYPE classType IDENT
            (^('extends' parent=type))?
            (^('implements' type+))?
            (^((FUNCTION_METHOD_DECL | ENTRY_FUNCTION_DECL | PRIMITIVE_VAR_DECLARATION |
                OBJECT_VAR_DECLARATION | CONSTRUCTOR_DECL | ENTRY_CONSTRUCTOR_DECL) .*))*)
        { currentClass = null; }
    ;

varDeclaration
    :   ^((PRIMITIVE_VAR_DECLARATION | OBJECT_VAR_DECLARATION)
            (^(MODIFIER_LIST .*))? type
            ^(VAR_DECLARATOR_LIST (^(VAR_DECLARATOR ^(IDENT .*) .*)
            )+))
    |   ^(FORMAL_PARAM_STD_DECL (^(MODIFIER_LIST .*))? type ^(IDENT .*))
        {
            $IDENT.def.type = $type.sym;
            $IDENT.symbolType = $type.sym;
        }
    ;

parenthesizedExpression
    :   ^(PAREN_EXPR exp=expression)
    ;

expression 
    :   ^(EXPR .*)
    ;

assignment
    :   ^(ASSIGNMENT IDENT expr)
    ;


type
    :   VOID {
            scope = $VOID.scope;
            typeText.add(new TypeName("void"));
        }
    |   ^(SIMPLE_TYPE t=. {
            scope = $SIMPLE_TYPE.scope;
            typeText.add(new TypeName($t.getText()));
        } .*)
    |   ^(OBJECT_TYPE { scope = $OBJECT_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT (^(TEMPLATE_INST
                (t1=type {tparams.add($t1.sym);} | lit1=literalVal {tparams.add($lit1.type);} )*))?
                {typeText.add(new TypeName($IDENT.text, tparams));}))+) .*)
    |   ^(REFERENCE_TYPE { scope = $REFERENCE_TYPE.scope; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT  {typeText.add(new TypeName($IDENT.text));} .*))+) .*)
    |   ^(PROXY_TYPE { scope = $PROXY_TYPE.scope; proxy = true; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT {typeText.add(new TypeName($IDENT.text));} .*))+) .*)
    |   ^(POINTER_TYPE { scope = $POINTER_TYPE.scope; pointer = true; }
            ^(QUALIFIED_TYPE_IDENT (^(IDENT (^(TEMPLATE_INST
            (t1=type {tparams.add($t1.sym);} | lit1=literalVal {tparams.add($lit1.type);} )*))?
            {typeText.add(new TypeName($IDENT.text, tparams));}))+) .*)
    ;

classType
    :   CLASS
    |   CHARE
    |   GROUP
    |   NODEGROUP
    |   MAINCHARE
    |   ^(CHARE_ARRAY ARRAY_DIMENSION)
    ;

block returns [boolean containsSDAG]
@init { boolean hasSDAGstatement = false; }
    :   ^(BLOCK (sdagNonLeader |
            sdagLeader {hasSDAGstatement = $sdagLeader.containsSDAG; } |
            block {hasSDAGstatement |= $block.containsSDAG})*) {
            $containsSDAG = hasSDAGstatement;
        }
    ;
    
statement
    :   sdagNonLeader
    |   sdagLeader
    |   block
    ;

sdagLeader returns [boolean containsSDAG]
    :   ^(OVERLAP block) {$containsSDAG = true;}
    |   ^(WHEN (IDENT expression? formalParameterList)* block) {$containsSDAG = true;}
    |   ^(IF parenthesizedExpression ifblock=block elseblock=block?) {
            $containsSDAG = $ifblock.containsSDAG || $elseblock.containsSDAG;
        }
    |   ^(FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block) {
            $containsSDAG = $block.containsSDAG;
        }
        -> {$containsSDAG}? ^(SDAG_FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block)
        -> ^(FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block)
    |   ^(FOR_EACH localModifierList? type IDENT expression block) {
            $containsSDAG = $block.containsSDAG;
        }
    |   ^(WHILE parenthesizedExpression block) {
            $containsSDAG = $block.containsSDAG;
        }
        -> {$containsSDAG}? ^(SDAG_WHILE parenthesizedExpression block)
        -> ^(WHILE parenthesizedExpression block)
    |   ^(DO block parenthesizedExpression) {
            $containsSDAG = $block.containsSDAG;
        }
        -> {$containsSDAG}? ^(SDAG_DO block parenthesizedExpression)
        -> ^(DO block parenthesizedExpression)
    ;

sdagNonLeader
    :   ^(PRIMITIVE_VAR_DECLARATION .*)
    |   ^(OBJECT_VAR_DECLARATION .*)
    |   ^(ASSERT expression expression?)
    |   ^(SWITCH parenthesizedExpression switchCaseLabel*)
    |   ^(RETURN expression?)
    |   ^(THROW expression)
    |   ^(BREAK IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled break not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(CONTINUE IDENT?) {
            if ($IDENT != null) {
                translator.error(this, "Labeled continue not supported yet, ignoring.", $IDENT);
            }
        }
    |   ^(LABELED_STATEMENT IDENT statement)
    |   expression
    |   ^('delete' expression)
    |   ^(EMBED STRING_LITERAL EMBED_BLOCK)
    |   ';' // Empty statement.
    |   ^(PRINT expression*)
    |   ^(PRINTLN expression*)
    |   ^(EXIT expression?)
    |   EXITALL
    ;
        
switchCaseLabel
    :   ^(CASE expression blockStatement*)
    |   ^(DEFAULT blockStatement*)
    ;
    
forInit
    :   localVariableDeclaration 
    |   expression+
    ;

