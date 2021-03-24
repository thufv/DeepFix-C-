using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      /* inserted */
      int _16 = 18;
      Console.WriteLine("Hello World!");
      Console.WriteLine("Press any key to exit.")
      /* ERROR: <missing ';'> */
      Console.ReadKey();
    }
  }
}
