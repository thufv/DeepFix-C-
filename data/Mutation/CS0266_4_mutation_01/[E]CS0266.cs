class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    /* inserted */
    mc = 6;
    DerivedClass dc = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
