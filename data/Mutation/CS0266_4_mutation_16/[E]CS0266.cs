class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    /* inserted */
    int _24 = 15;
    DerivedClass dc = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
