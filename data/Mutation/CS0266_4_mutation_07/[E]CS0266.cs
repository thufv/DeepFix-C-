class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    DerivedClass dc = new DerivedClass();
    /* inserted */
    int _22 = 2;
    dc = mc;
  }
}
class DerivedClass: MyClass {}
