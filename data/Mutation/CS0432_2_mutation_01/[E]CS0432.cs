public class B {
  public static void Meth() {}
}
public class Test {
  public static void Main() {
    /* inserted */
    int _28 = 2;
    B::Meth();
  }
}
