using System;
class Program {
  static void Main() {
    try {
      /* inserted */
      int _25 = 10;
    } catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
