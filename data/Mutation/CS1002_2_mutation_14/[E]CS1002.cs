using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!")
      /* ERROR: <missing ';'> */
      /* inserted */
      int _13 = 20;
      Console.WriteLine("Press any key to exit.");
      Console.ReadKey();
    }
  }
}
