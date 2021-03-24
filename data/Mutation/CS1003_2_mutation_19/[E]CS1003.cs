using System;
namespace Loops {
  class Program {
    static void Main(string[] args) {
      for
      /* ERROR: <missing '('> */
      ;;) {
        /* inserted */
        int _30 = 27;
        Console.WriteLine("Hey! I am Trapped");
      }
    }
  }
}
