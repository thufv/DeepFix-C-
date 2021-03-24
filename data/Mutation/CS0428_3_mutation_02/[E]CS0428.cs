namespace ConsoleApplication1 {
  class Program {
    delegate int Del1();
    delegate object Del2();
    static void Main(string[] args) {
      ExampleClass ec = new ExampleClass();
      ec = ExampleClass.Method2;
    }
  }
  public class ExampleClass {
    public int Method1() {
      return 0
      /* updated */
      ;
    }
    public static ExampleClass Method2() {
      return null;
    }
  }
}
