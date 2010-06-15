package charj.translator;

public class ASTHTMLPrinter implements TreeTraverser.Visitor {
    StringBuilder out_;

    public ASTHTMLPrinter() {
        out_ = new StringBuilder(1024);
    }

    public void setUp(CharjAST tree) {
        out_.append("<html><head>\n");
        out_.append("<meta http-equiv=\"content-type\" ");
        out_.append("content=\"text/html; charset=utf-8\">");
        out_.append("<title>Charj AST</title>\n");
        out_.append("<link rel=\"stylesheet\" type=\"text/css\" \n");
        out_.append("href=\"http://yui.yahooapis.com/combo?2.7.0/build/fonts/");
        out_.append("fonts-min.css&2.7.0/build/base/base-min.css&2.7.0/build/");
        out_.append("treeview/assets/skins/sam/treeview.css\">\n");
        out_.append("<script type=\"text/javascript\" ");
        out_.append("src=\"http://yui.yahooapis.com/combo?2.7.0/build/");
        out_.append("yahoo-dom-event/yahoo-dom-event.js&2.7.0/build/animation/");
        out_.append("animation-min.js&2.7.0/build/treeview/treeview-min.js\">");
        out_.append("</script></head><body><div id=\"ASTTree\"><ol>\n");
    }

    public void tearDown(CharjAST tree) {
        out_.append("</ol></div><script type=\"text/javascript\">\n");
        out_.append("var tree = new YAHOO.widget.TreeView(\"ASTTree\");\n");
        out_.append("tree.expandAll()\n");
        out_.append("tree.draw()\n");
        out_.append("</script></body></html>\n");
    }

    public void visit(CharjAST tree,
            CharjAST parent,
            int childIndex,
            int visitType)
    {
        switch(visitType) {
            case VISIT_ENTER:
                out_.append("<li>" + tree.toString());
                if (tree.symbolType != null) {
                    out_.append(" [type=" + tree.symbolType + "]");
                    out_.append(" [def=" + tree.def + "]");
                }
                if (tree.getChildCount() != 0) {
                    out_.append("<ol>");
                } else {
                    out_.append("</li>");
                }
                break;
            case VISIT_EXIT:
                if (tree.getChildCount() != 0) {
                    out_.append("</ol></li>");
                }
                break;
        }
    }

    public String output() {
        return out_.toString();
    }
}
