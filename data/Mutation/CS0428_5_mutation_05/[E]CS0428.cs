using System;
class Program {
  static void Main() {
    try {} catch(Exception ex) {
      /* inserted */
      int _25 = 22;
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
