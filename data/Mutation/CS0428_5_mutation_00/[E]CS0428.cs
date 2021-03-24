using System;
class Program {
  static void Main() {
    /* inserted */
    int _5 = 2;
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
