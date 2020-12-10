using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!");
      Console.WriteLine("Press any key to exit. "
      /* updated */
      );
      Console.ReadKey()
      /* ERROR: <missing ';'> */
    }
  }
}
