using System;
namespace Loops {
  class Program {
    static void Main(string[] args) {
      for
      /* ERROR: <missing '('> */
      ;;) {
        Console.WriteLine("_Hey! I am Trapped"
        /* updated */
        );
      }
    }
  }
}
