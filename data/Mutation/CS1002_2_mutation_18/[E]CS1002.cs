using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!")
      /* ERROR: <missing ';'> */
      /* inserted */
      int _22 = 5;
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey();
    }
  }
}
