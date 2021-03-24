using System;
namespace Loops {
  class Program {
    static void Main(string[] args) {
      int a = 10;
      while
      /* ERROR: <missing '('> */
      a < 22
      /* updated */
      ) {
        Console.WriteLine("value of a: {0}", a);
        a++;
      }
      Console.ReadLine();
    }
  }
}
