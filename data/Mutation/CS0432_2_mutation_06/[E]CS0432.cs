public class B {
  public static void Meth() {
    /* inserted */
    int _23 = 28;
  }
}
public class Test {
  public static void Main() {
    B::Meth();
  }
}
