public class MyClass {
  private void MyMethod() {
    /* inserted */
    int _20 = 24;
  }
}
public class MyClass2 {
  public static int Main() {
    MyClass a = new MyClass();
    a.MyMethod();
    return 0;
  }
}
