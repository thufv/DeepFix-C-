using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("_Hello World!"
      /* updated */
      )
      /* ERROR: <missing ';'> */
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey();
    }
  }
}
