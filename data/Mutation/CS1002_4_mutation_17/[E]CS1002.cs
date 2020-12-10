using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      /* inserted */
      int _25 = 25;
      Console.WriteLine("Hello World!");
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey()
      /* ERROR: <missing ';'> */
    }
  }
}
