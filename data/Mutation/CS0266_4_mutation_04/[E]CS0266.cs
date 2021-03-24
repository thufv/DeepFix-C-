class MyClass {
  public static void Main() {
    /* inserted */
    int _10 = 9;
    MyClass mc = new MyClass();
    DerivedClass dc = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
