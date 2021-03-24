using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!")
      /* ERROR: <missing ';'> */
      Console.WriteLine("Press any key to exit.");
      /* inserted */
      int _27 = 9;
      Console.ReadKey();
    }
  }
}
