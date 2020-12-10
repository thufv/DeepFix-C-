namespace A {
  class Program {
    static void Main() {
      /* inserted */
      int _21 = 13;
      Test test = new Test(1);
    }
  }
  class Test {
    private Test(int i) {}
  }
}
