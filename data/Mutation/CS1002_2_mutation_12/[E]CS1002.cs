using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!")
      /* ERROR: <missing ';'> */
      /* inserted */
      int _7 = 10;
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey();
    }
  }
}
