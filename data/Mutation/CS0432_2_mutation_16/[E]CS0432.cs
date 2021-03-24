public class B {
  public static void Meth() {
    /* inserted */
    int _1 = 18;
  }
}
public class Test {
  public static void Main() {
    B::Meth();
  }
}
