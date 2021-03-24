public class B {
  public static void Meth() {
    /* inserted */
    int _13 = 20;
  }
}
public class Test {
  public static void Main() {
    B::Meth();
  }
}
