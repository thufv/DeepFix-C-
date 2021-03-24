using System;
namespace Loops {
  class Program {
    static void Main(string[] args) {
      int a = 10;
      /* inserted */
      int _5 = 24;
      while
      /* ERROR: <missing '('> */
      a < 20) {
        Console.WriteLine("value of a: {0}", a);
        a++;
      }
      Console.ReadLine();
    }
  }
}
