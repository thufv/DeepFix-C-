using System;
class Program {
  static void Main() {
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      /* inserted */
      int _23 = 26;
      Console.WriteLine(MyException);
    }
  }
}
