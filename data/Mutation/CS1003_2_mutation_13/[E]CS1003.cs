using System;
namespace Loops {
  class Program {
    static void Main(string[] args) {
      for
      /* ERROR: <missing '('> */
      ;;) {
        /* inserted */
        int _15 = 26;
        Console.WriteLine("Hey! I am Trapped");
      }
    }
  }
}
