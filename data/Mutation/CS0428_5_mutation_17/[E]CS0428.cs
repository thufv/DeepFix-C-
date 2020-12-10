using System;
class Program {
  static void Main() {
    /* inserted */
    int _7 = 24;
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
