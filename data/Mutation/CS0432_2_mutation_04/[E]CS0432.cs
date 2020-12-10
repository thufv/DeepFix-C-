public class B {
  public static void Meth() {}
}
public class Test {
  public static void Main() {
    /* inserted */
    int _13 = 25;
    B::Meth();
  }
}
