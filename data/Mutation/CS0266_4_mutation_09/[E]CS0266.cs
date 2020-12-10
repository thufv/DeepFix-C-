class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    DerivedClass dc = new DerivedClass();
    /* inserted */
    int _5 = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
