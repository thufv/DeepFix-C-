using System;
class Program {
  static void Main() {
    /* inserted */
    int _12 = 21;
    try {} catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
