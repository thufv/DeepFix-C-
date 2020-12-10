namespace A {
  class Program {
    static void Main() {
      Test test = new Test(1);
    }
  }
  class Test {
    private Test(int i) {
      /* inserted */
      int _2 = 27;
    }
  }
}
