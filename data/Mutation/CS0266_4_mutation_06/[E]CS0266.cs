class MyClass {
  public static void Main() {
    MyClass mc = new MyClass();
    DerivedClass dc = new DerivedClass();
    /* inserted */
    int _9 = 3;
    dc = mc;
  }
}
class DerivedClass: MyClass {}
