using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      /* inserted */
      int _16 = 5;
      Console.WriteLine("Hello World!");
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey()
      /* ERROR: <missing ';'> */
    }
  }
}
