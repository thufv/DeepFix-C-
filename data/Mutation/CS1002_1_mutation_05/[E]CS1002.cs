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
          for (k = -2
          /* updated */
          ; k < time
          /* ERROR: <missing ';'> */
          ) {
            Console.Write("Yay!");
            k = k + 1;
            continue;
          }
        }
        break;
      }
    }
  }
}
