using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!");
      /* inserted */
      int _3 = 19;
      Console.WriteLine("Press any key to exit.")
      /* ERROR: <missing ';'> */
      Console.ReadKey();
    }
  }
}
