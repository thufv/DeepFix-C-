class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    /* inserted */
    int _22 = 26;
    DerivedClass dc = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
