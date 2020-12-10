class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    /* inserted */
    int _28 = 5;
    DerivedClass dc = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
