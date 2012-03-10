
package charj.translator;

import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;

public class TypeName {
    public String name;
    public List<Type> parameters;

    public TypeName(String _name, List<Type> _parameters) {
        name = _name;
        parameters = _parameters;
    }

    public TypeName(String _name) {
        name = _name;
        parameters = null;
    }

    public String toString() {
        String params = "";
        if (parameters != null) {
            StringBuilder sb = new StringBuilder();
            Iterator<Type> it = parameters.iterator();
            while (it.hasNext()) {
                sb.append(it.next().toString());
                sb.append(",");
            }
            params = sb.toString();
        }
        if (params != "") {
          return name + "<" + params + ">";
        } else {
          return name;
        }
    }

    public static List<TypeName> createTypeName(String name) {
        List<TypeName> list = new ArrayList<TypeName>();
        list.add(new TypeName(name));
        return list;
    }

    public static String typeToString(List<TypeName> type) {
        if (type != null && type.size() != 0) {
            StringBuilder sb = new StringBuilder();
            Iterator<TypeName> it = type.iterator();
            while (it.hasNext()) {
                sb.append(it.next().toString());
            }
            return sb.toString();
        } else {
            return "";
        }
    }
}
