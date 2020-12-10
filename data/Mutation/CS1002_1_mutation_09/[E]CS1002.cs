using System;
namespace Treehouse.CodeChallenges {
  class Program {
    static void Main() {
      while (true) {
        Console.Write("Enter the number of times to print \"Yay!\": ");
        string entry = Console.ReadLine();
        if (entry.ToLower() == "quit") {
          break;
        } else {
          int time = int.Parse(entry);
          int k;
          for (k = 0; k < time
          /* ERROR: <missing ';'> */
          ) {
            Console.Write("Yay!");
            k = k + 2
            /* updated */
            ;
            continue;
          }
        }
        break;
      }
    }
  }
}
