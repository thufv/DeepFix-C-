class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    /* inserted */
    int _3 = 4;
    DerivedClass dc = new DerivedClass();
    dc = mc;
  }
}
class DerivedClass: MyClass {}
