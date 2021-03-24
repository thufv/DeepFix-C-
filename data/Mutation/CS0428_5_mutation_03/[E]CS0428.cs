using System;
class Program {
  static void Main() {
    /* inserted */
    int _25 = 23;
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
