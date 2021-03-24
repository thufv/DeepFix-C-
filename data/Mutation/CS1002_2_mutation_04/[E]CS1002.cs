using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!_"
      /* updated */
      )
      /* ERROR: <missing ';'> */
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey();
    }
  }
}
