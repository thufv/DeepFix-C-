using System;
namespace Loops {
  class Program {
    static void Main(string[] args) {
      int a = 9
      /* updated */
      ;
      do {
        Console.WriteLine("value of a: {0}", a);
        a = a + 1;
      } while
      /* ERROR: <missing '('> */
      a < 20 );
      Console.ReadLine();
    }
  }
}
