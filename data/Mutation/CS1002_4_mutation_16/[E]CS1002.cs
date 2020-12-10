using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!");
      Console.WriteLine("Press any key to exit.");
      /* inserted */
      int _10 = 13;
      Console.ReadKey()
      /* ERROR: <missing ';'> */
    }
  }
}
