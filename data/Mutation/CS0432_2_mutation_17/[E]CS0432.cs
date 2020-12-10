public class B {
  public static void Meth() {}
}
public class Test {
  public static void Main() {
    /* inserted */
    int _9 = 27;
    B::Meth();
  }
}
