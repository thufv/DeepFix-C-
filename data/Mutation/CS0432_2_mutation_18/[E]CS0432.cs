public class B {
  public static void Meth() {
    /* inserted */
    int _3 = 10;
  }
}
public class Test {
  public static void Main() {
    B::Meth();
  }
}
