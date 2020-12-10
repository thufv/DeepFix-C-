using System;
class Program {
  static void Main() {
    try {
      /* inserted */
      int _24 = 26;
    } catch(Exception ex) {
      string MyException = ex.ToString;
      Console.WriteLine(MyException);
    }
  }
}
