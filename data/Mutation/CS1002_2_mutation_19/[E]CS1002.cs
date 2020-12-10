using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!")
      /* ERROR: <missing ';'> */
      Console.WriteLine("Press any key to exit.");
      /* inserted */
      int _28 = "Press any key to exit.";
      Console.ReadKey();
    }
  }
}
