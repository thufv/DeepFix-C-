using System;
class Program {
  static void Main() {
    /* inserted */
    int _21 = 1;
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
