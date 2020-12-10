class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    DerivedClass dc = new DerivedClass();
    /* inserted */
    int _26 = 20;
    dc = mc;
  }
}
class DerivedClass: MyClass {}
