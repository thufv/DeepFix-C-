class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    DerivedClass dc = new DerivedClass();
    /* inserted */
    int _18 = 21;
    dc = mc;
  }
}
class DerivedClass: MyClass {}
