using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!");
      /* inserted */
      int _30 = 7;
      Console.WriteLine("Press any key to exit.")
      /* ERROR: <missing ';'> */
      Console.ReadKey();
    }
  }
}
