package charj.translator;

public abstract class TreeTraverser {

    public interface Visitor {
        public static final int VISIT_ENTER = 0;
        public static final int VISIT_EXIT = 1;

        public void setUp(CharjAST tree);
        public void tearDown(CharjAST tree);
        public void visit(CharjAST tree,
                CharjAST parent,
                int childIndex,
                int visitType);
    }

    /** Execute the visit action on each tree node. */
    public static void visit(CharjAST tree, Visitor visitor) {
        visitor.setUp(tree);
        _visit(tree, null, 0, visitor);
        visitor.tearDown(tree);
    }

    /** Do the recursive tree walk for a visit */
    protected static void _visit(CharjAST tree,
            CharjAST parent,
            int childIndex,
            Visitor visitor)
    {
        if (tree == null) return;
        visitor.visit(tree, parent, childIndex, Visitor.VISIT_ENTER);
        int n = tree.getChildCount();
        for (int i=0; i<n; ++i) {
            CharjAST child = (CharjAST)tree.getChild(i);
            _visit(child, tree, i, visitor);
        }
        visitor.visit(tree, parent, childIndex, Visitor.VISIT_EXIT);
    }
}
