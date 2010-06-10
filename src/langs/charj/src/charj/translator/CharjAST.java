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
    public Symbol symbol;
    
    public CharjAST(Token t) {
        super(t);
    }

    public CharjAST(int type, String text) {
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
        try
        {
            return (CharjAST) super.getChild(index);
        }
        catch(ClassCastException e)
        {
            Tree child = super.getChild(index);
            System.out.println("possible error node: " + child);
            return new CharjAST(child.getType(), child.getText());
        }
    }
    
    @Override
    public String toString() {
        String s = super.toString();
        if (symbol != null) {
            s += "(" + symbol + ")" ;
        }
        return s;
    }
    
    @Override
    public CharjAST dupNode()
    {
        CharjAST node = new CharjAST(getType(), getText());
        node.symbol = symbol;
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
        try
        {
            for(CharjAST c : getChildren())
                if(c.getType() == type)
                    return c;
        }
        catch(NullPointerException npe)
        {
            npe.printStackTrace();
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

    @Override
    public boolean equals(Object o)
    {
        if(o == null)
            return false;
        if(!(o instanceof CharjAST))
            return false;
        CharjAST other = (CharjAST)o;
        return other.getType() == this.getType() && other.getText().equals(this.getText());
    }

    public void insertChild(int index, CharjAST node)
    {
        try
        {
            List<CharjAST> children = new ArrayList<CharjAST>();

            for(int i=0; i<index; i++)
                children.add(getChild(i));
                
            children.add(node);

            for(int i=index; i < getChildCount(); i++)
                children.add(getChild(i));

            getChildren().clear();

            for(CharjAST c : children)
               addChild(c);
        }
        catch(NullPointerException npe)
        {
            //npe.printStackTrace();
            if(index == 0)
                addChild(node);
            else
                throw new ArrayIndexOutOfBoundsException(index);
        }
    }

    public void setType(int type, String name)
    {
        try
        {
            Field tokenField = getClass().getSuperclass().getDeclaredField("token");
            tokenField.set(this, new CommonToken(type, name));
        }
        catch(Exception e)
        {
            System.err.println("debugging, this should never happen");
            e.printStackTrace();
        }
    }
}
