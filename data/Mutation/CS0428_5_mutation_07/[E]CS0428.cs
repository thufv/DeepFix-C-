using System;
class Program {
  static void Main() {
    try {
      /* inserted */
      int _14 = 24;
    } catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
