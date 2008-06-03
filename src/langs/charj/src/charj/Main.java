package charj;

import charj.translator.Translator;
import java.io.FileInputStream;

public class Main 
{
    public static void main(String[] args) throws Exception
    {
        System.out.println(Translator.translate(new FileInputStream(args[0])));
    }
}
