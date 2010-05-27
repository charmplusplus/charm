package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;

class PupRoutineCreator
{
    private CharjAST pupNode;
    private CharjAST initNode;

    PupRoutineCreator()
    {
        createPupNode();
        createInitNode();
    }

    protected CharjAST getPupRoutineNode()
    {
        return pupNode;
    }

    protected CharjAST getInitRoutineNode()
    {
        return initNode;
    }

    private CharjAST createNode(int type, String text)
    {
        return new CharjAST(new CommonToken(type, text));
    }    
    
    private void createInitNode()
    {
        initNode = createNode(CharjParser.FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL");

    	initNode.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
        initNode.addChild(createNode(CharjParser.VOID, "void"));
        initNode.addChild(createNode(CharjParser.IDENT, "initMethod"));
        initNode.addChild(createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST"));
        initNode.addChild(createNode(CharjParser.BLOCK, "BLOCK"));

        initNode.getChild(0).addChild(createNode(CharjParser.PRIVATE, "private"));
    }

    private void createPupNode()
    {
        pupNode = createNode(CharjParser.FUNCTION_METHOD_DECL, "FUNCTION_METHOD_DECL");

    	pupNode.addChild(createNode(CharjParser.MODIFIER_LIST, "MODIFIER_LIST"));
        pupNode.addChild(createNode(CharjParser.VOID, "void"));
        pupNode.addChild(createNode(CharjParser.IDENT, "pup"));
        pupNode.addChild(createNode(CharjParser.FORMAL_PARAM_LIST, "FORMAL_PARAM_LIST"));
        pupNode.addChild(createNode(CharjParser.BLOCK, "BLOCK"));

        pupNode.getChild(0).addChild(createNode(CharjParser.PUBLIC, "public"));

        pupNode.getChild(3).addChild(createNode(CharjParser.FORMAL_PARAM_STD_DECL, "FORMAL_PARAM_STD_DECL"));
        pupNode.getChild(3).getChild(0).addChild(createNode(CharjParser.REFERENCE_TYPE, "REFERENCE_TYPE"));
        pupNode.getChild(3).getChild(0).getChild(0).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
        pupNode.getChild(3).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "PUP::er"));

        pupNode.getChild(3).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
    }

    protected CharjAST getEnclosingType(CharjAST varDeclNode)
    {
        for(CharjAST p = varDeclNode.getParent(); p != null; p = p.getParent())
            if(p.getType() == CharjParser.TYPE)
                return p;
        return null;
    }         

    protected void varPup(CharjAST idNode)
    {
        System.out.println("in var pup: " + idNode.getText());
        int type = -1;

        for(CharjAST p = idNode.getParent(); p != null; p = p.getParent())
        {
            switch(p.getType())
            {
                case CharjParser.PRIMITIVE_VAR_DECLARATION:
                    System.out.println("got the type, it's primitive");
                    type = p.getType();
                    break;
                case CharjParser.OBJECT_VAR_DECLARATION:
                    System.out.println("got the type, it's an object!");
                    type = p.getChild(0).getType();
                    break;
                case CharjParser.FUNCTION_METHOD_DECL:
                case CharjParser.BLOCK:
                case CharjParser.FORMAL_PARAM_LIST:
                    System.out.println("local var, not puping...");
                    return;
                case CharjParser.TYPE:
                    System.out.println("class member var, puping... type " + type);
                    switch(type)
                    {
                        case CharjParser.REFERENCE_TYPE: System.out.println("found ref");
                        case CharjParser.PRIMITIVE_VAR_DECLARATION: System.out.println("found simple or ref");
                            primitiveVarPup(idNode);
                            break;
                        case CharjParser.POINTER_TYPE: System.out.println("found pointer");
                            pointerVarPup(idNode);
                            break;
                        default:
                            System.out.println("unknown type -- THIS IS AN ERROR in method varPup, in PupRoutineCreator.java");
                            break;
                    }
                    return;
            }
        }
        System.out.println("THIS IS AN ERROR in method varPup, in PupRoutineCreator.java");
    }

    protected void primitiveVarPup(CharjAST idNode)
    {
        pupNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));
        
        int index = pupNode.getChild(4).getChildren().size() - 1;

        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.BITWISE_OR, "|"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(idNode.dupNode());
    }
    
    private boolean generatedIf = false;

    protected void pointerVarPup(CharjAST idNode)
    {
        if(!generatedIf)
        {
            generateIf();
            generatedIf = true;
        }

        // add stuff to the initMethod routine
        initNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));

        int index = initNode.getChild(4).getChildren().size() - 1;

        initNode.getChild(4).getChild(index).addChild(createNode(CharjParser.ASSIGNMENT, "="));
        initNode.getChild(4).getChild(index).getChild(0).addChild(idNode.dupNode());
        initNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.NEW, "new"));
        initNode.getChild(4).getChild(index).getChild(0).getChild(1).addChild(createNode(CharjParser.QUALIFIED_TYPE_IDENT, "QUALIFIED_TYPE_IDENT"));
        initNode.getChild(4).getChild(index).getChild(0).getChild(1).getChild(0).addChild(idNode.getParent().getParent().getParent().getChild(0).getChild(0).getChild(0).dupNode());
        initNode.getChild(4).getChild(index).getChild(0).getChild(1).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));

        // add stuff to the pup routine
        pupNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));

        index = pupNode.getChild(4).getChildren().size() - 1;

        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.ARROW, "ARROW"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(idNode.dupNode());
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "pup"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(1).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(1).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
    }

    protected void generateIf()
    {
        pupNode.getChild(4).addChild(createNode(CharjParser.IF, "if"));
        
        int index = pupNode.getChild(4).getChildren().size() - 1;
       
        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.PAREN_EXPR, "PAREN_EXPR"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.DOT, "."));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "isUnpacking"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));
        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.BLOCK, "BLOCK"));
        pupNode.getChild(4).getChild(index).getChild(1).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(index).getChild(1).getChild(0).addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        pupNode.getChild(4).getChild(index).getChild(1).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "initMethod"));
        pupNode.getChild(4).getChild(index).getChild(1).getChild(0).getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));
    }

}
