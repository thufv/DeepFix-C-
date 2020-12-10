using System;
class Program {
  static void Main() {
    /* inserted */
    int _7 = 5;
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
