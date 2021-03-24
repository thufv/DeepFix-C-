using System;
class Program {
  static void Main() {
    try {
      /* inserted */
      int _7 = 14;
    } catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
