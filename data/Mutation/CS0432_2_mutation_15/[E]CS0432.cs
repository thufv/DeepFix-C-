public class B {
  public static void Meth() {}
}
public class Test {
  public static void Main() {
    /* inserted */
    int _17 = 15;
    B::Meth();
  }
}
