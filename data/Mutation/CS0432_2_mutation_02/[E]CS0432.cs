public class B {
  public static void Meth() {}
}
public class Test {
  public static void Main() {
    /* inserted */
    int _6 = 9;
    B::Meth();
  }
}
