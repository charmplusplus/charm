
tree grammar AtomicBlocks;

options {
    tokenVocab = Charj;
    ASTLabelType = CharjAST;
    filter = true;
    output = AST;
}

@header {
package charj.translator;
import java.util.Iterator;
}

@members {
    SymbolTable symtab;

    public AtomicBlocks(TreeNodeStream input, SymbolTable symtab) {
        this(input);
        this.symtab = symtab;
    }
}


topdown
    :   block
    ;


parenthesizedExpression
    :   ^(PAREN_EXPR exp=expression)
    ;

expression 
    :   ^(EXPR .*)
    ;

assignment
    :   ^(ASSIGNMENT IDENT expression)
    ;


type
    :   VOID 
    |   ^(SIMPLE_TYPE .*)
    |   ^(OBJECT_TYPE .*)
    |   ^(REFERENCE_TYPE .*)
    |   ^(PROXY_TYPE .*)
    |   ^(POINTER_TYPE .*)
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
            b=block {hasSDAGstatement |= $b.containsSDAG;})*) {
            $containsSDAG = hasSDAGstatement;
            ((LocalScope)$BLOCK.def).hasSDAG = $containsSDAG;
        }
    ;
    
statement
    :   sdagNonLeader
    |   sdagLeader
    |   block
    ;

sdagLeader returns [boolean containsSDAG]
@init { $containsSDAG = false; }
    :   ^(OVERLAP block) {$containsSDAG = true;}
    |   ^(WHEN (IDENT expression? ^(FORMAL_PARAM_LIST .*))* block) {$containsSDAG = true;}
    |   ^(IF parenthesizedExpression ifblock=block elseblock=block?
            {$ifblock.containsSDAG || $elseblock.containsSDAG}?)
        {
            $containsSDAG = true;
        }
        -> ^(SDAG_IF parenthesizedExpression $ifblock $elseblock)
    |   ^(FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block {$block.containsSDAG}?) {
            $containsSDAG = true;
        }
        -> ^(SDAG_FOR forInit? FOR_EXPR expression? FOR_UPDATE expression* block)
    |   ^(WHILE parenthesizedExpression block {$block.containsSDAG}?) {
            $containsSDAG = true;
        }
        -> ^(SDAG_WHILE parenthesizedExpression block)
    |   ^(DO block parenthesizedExpression {$block.containsSDAG}?) {
            $containsSDAG = true;
        }
        -> ^(SDAG_DO block parenthesizedExpression)
    ;

sdagNonLeader
    :   ^(PRIMITIVE_VAR_DECLARATION .*)
    |   ^(OBJECT_VAR_DECLARATION .*)
    |   ^(ASSERT expression expression?)
    |   ^(SWITCH parenthesizedExpression switchCaseLabel*)
    |   ^(RETURN expression?)
    |   ^(THROW expression)
    |   ^(BREAK IDENT?)
    |   ^(CONTINUE IDENT?)
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
    :   ^(CASE expression statement*)
    |   ^(DEFAULT statement*)
    ;
    
forInit
    :   ^(PRIMITIVE_VAR_DECLARATION .*)
    |   ^(OBJECT_VAR_DECLARATION .*)
    |   expression+
    ;

