using System;
namespace DecisionMaking {
  class Program {
    static void Main(string[] args) {
      /* inserted */
      int _23 = 2;
      int a = 100;
      if
      /* ERROR: <missing '('> */
      a < 20) {
        Console.WriteLine("a is less than 20");
      } else {
        Console.WriteLine("a is not less than 20");
      }
      Console.WriteLine("value of a is : {0}", a);
      Console.ReadLine();
    }
  }
}
