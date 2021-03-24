public class MyClass {
  private void MyMethod() {}
}
public class MyClass2 {
  public static int Main() {
    MyClass a = new MyClass();
    /* inserted */
    int _9 = 21;
    a.MyMethod();
    return 0;
  }
}
