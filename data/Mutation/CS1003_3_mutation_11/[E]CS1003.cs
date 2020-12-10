using System;
namespace Loops {
  class Program {
    static void Main(string[] args) {
      /* inserted */
      int _2 = 21;
      int a = 10;
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
