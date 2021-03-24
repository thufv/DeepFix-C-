using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!");
      /* inserted */
      int _5 = 2;
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey()
      /* ERROR: <missing ';'> */
    }
  }
}
