package charj.translator;

import org.antlr.runtime.tree.CommonTree;
import org.antlr.runtime.tree.Tree;
import org.antlr.runtime.Token;
import org.antlr.runtime.CommonToken;
import java.util.*;
import java.lang.reflect.*;

/**
 * Custom subclass of Antlr's tree node. Doesn't do anything special yet,
 * it's just here to make it easier to add features later.
 */
public class CharjAST extends CommonTree
{
    /** Semantic information about this node. Includes type, scope, location
     * of definition, etc. */
    public Scope scope;
    public Symbol def;
    public Type symbolType;
    
    public CharjAST(Token t)
	{
        super(t);
    }

    public CharjAST(int type, String text)
	{
        super(new CommonToken(type, text));
    }

    public CharjAST getParent()
    {
        return (CharjAST) super.getParent();
    }

    public List<CharjAST> getChildren()
    {
        return (List<CharjAST>) super.getChildren();
    }

    public CharjAST getChild(int index)
    {
        try {
            return (CharjAST) super.getChild(index);
        } catch(ClassCastException e) {
            Tree child = super.getChild(index);
            System.out.println("WARNING, possible error node: " + child);
            return new CharjAST(child.getType(), child.getText());
        }
    }
    
    @Override
    public String toString()
	{
        String s = super.toString();
        if (symbolType != null) {
            s += "(" + symbolType + ")" ;
        }
        return s;
    }
    
    @Override
    public CharjAST dupNode()
    {
        CharjAST node = new CharjAST(getType(), getText());
        node.def = def;
        node.scope = scope;
        node.symbolType = symbolType;
        return node;
    }

    public CharjAST dupTree()
    {
        CharjAST root = dupNode();

        List<CharjAST> children = getChildren();
        if(children == null)
            return root;

        for(CharjAST child : getChildren())
            root.addChild(child.dupTree());
        return root;
    }

    public CharjAST getChildOfType(int type)
    {
        List<CharjAST> children = getChildren();
        
        if (children != null) {
            for(CharjAST c : children)
                if(c.getType() == type)
                    return c;
        }
        
        return null;
    }

    public CharjAST getChildAfterType(int type)
    {
        for(int i = 0; i < getChildCount(); i++)
            if(getChild(i).getType() == type)
                return getChild(i+1);
        return getChild(0);
    }

    public boolean hasParentOfType(int type)
    {
        //System.out.println("checking parent type = " + type);
        CharjAST node = getParent();
        while (node != null && node.getType() != type) {
            //System.out.println("looking at parents, type = " + node.getType() + ": " + node.toString());
            node = node.getParent();
        }
        boolean found = (node != null);
        //if (found) System.out.println("looking at parents, type = " + node.getType() + ": " + node.toString());
        //else System.out.println("null parent");
        //System.out.println("Result: " + found);
        return found;
    }

    @Override
    public boolean equals(Object o)
    {
        if (o == null) return false;
        if (!(o instanceof CharjAST)) return false;
        CharjAST other = (CharjAST)o;
        return other.getType() == this.getType() &&
			other.getText().equals(this.getText());
    }

    public void insertChild(int index, CharjAST node)
    {
        try {
            List<CharjAST> children = new ArrayList<CharjAST>();

            for(int i=0; i<index; i++) children.add(getChild(i));
                
            children.add(node);

            for(int i=index; i < getChildCount(); i++) children.add(getChild(i));

            getChildren().clear();

            for(CharjAST c : children) addChild(c);
        } catch(NullPointerException npe) {
			//TODO: fix this bad code, do not catch an NPE and act on it
            //npe.printStackTrace();
            if(index == 0) addChild(node);
            else throw new ArrayIndexOutOfBoundsException(index);
        }
    }

    public void setType(int type, String name)
    {
        try {
            Field tokenField = getClass().getSuperclass().getDeclaredField("token");
            tokenField.set(this, new CommonToken(type, name));
        } catch(Exception e) {
            System.err.println("debugging, this should never happen");
            e.printStackTrace();
        }
    }

    // Does this subtree contain an accelerator modifier?
    public boolean hasAccelModifier()
    {
        return false;
    }

}
