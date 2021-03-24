using System;
class Program {
  static void Main() {
    /* inserted */
    int _10 = 10;
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
