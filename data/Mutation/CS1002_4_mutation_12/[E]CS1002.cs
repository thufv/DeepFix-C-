using System;
namespace HelloWorld {
  class Hello {
    static void Main() {
      Console.WriteLine("Hello World!");
      Console.WriteLine("Press any key to exit.");
      /* inserted */
      int _13 = 24;
      Console.ReadKey()
      /* ERROR: <missing ';'> */
    }
  }
}
