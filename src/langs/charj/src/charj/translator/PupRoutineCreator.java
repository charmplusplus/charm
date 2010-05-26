package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;

class PupRoutineCreator
{
    private CharjAST pupNode;
    private CharjAST block;

    PupRoutineCreator()
    {
        createPupNode();
        block = pupNode.getChild(4);
    }

    protected CharjAST getPupRoutineNode()
    {
        return pupNode;
    }

    private CharjAST createNode(int type, String text)
    {
        return new CharjAST(new CommonToken(type, text));
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
                    }
                    return;
            }
        }
        System.out.println("after for");
    }

    protected void primitiveVarPup(CharjAST idNode)
    {
        pupNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));
        
        int index = pupNode.getChild(4).getChildren().size() - 1;

        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.BITWISE_OR, "|"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(idNode.dupNode());
    }
    
    protected void pointerVarPup(CharjAST idNode)
    {
        pupNode.getChild(4).addChild(createNode(CharjParser.EXPR, "EXPR"));
        
        int index = pupNode.getChild(4).getChildren().size() - 1;

        pupNode.getChild(4).getChild(index).addChild(createNode(CharjParser.METHOD_CALL, "METHOD_CALL"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.DOT, "."));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(idNode.dupNode());
        pupNode.getChild(4).getChild(index).getChild(0).getChild(0).addChild(createNode(CharjParser.IDENT, "pup"));
        pupNode.getChild(4).getChild(index).getChild(0).addChild(createNode(CharjParser.ARGUMENT_LIST, "ARGUMENT_LIST"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(1).addChild(createNode(CharjParser.EXPR, "EXPR"));
        pupNode.getChild(4).getChild(index).getChild(0).getChild(1).getChild(0).addChild(createNode(CharjParser.IDENT, "p"));
    }

}
