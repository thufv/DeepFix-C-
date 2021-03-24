using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!");
      Console.WriteLine("Press any key to exit.")
      /* ERROR: <missing ';'> */
      /* inserted */
      int _7 = 2;
      Console.ReadKey();
    }
  }
}
