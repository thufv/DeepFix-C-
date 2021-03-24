public class MyClass {
  private void MyMethod() {}
}
public class MyClass2 {
  public static int Main() {
    MyClass a = new MyClass();
    a.MyMethod();
    /* inserted */
    int _19 = new MyClass();
    return 0;
  }
}
