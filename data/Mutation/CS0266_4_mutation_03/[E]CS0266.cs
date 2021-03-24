class MyClass {
  public static void Main() {
    /* inserted */
    int _22 = 1;
    MyClass mc = new MyClass();
    DerivedClass dc = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
