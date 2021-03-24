using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
namespace testscores {
  class Program {
    static void Main(string[] args) {
      string myChoice = "";
      /* inserted */
      int _15 = 16;
      switch (myChoice) {
      case "K":
      case "k":
        Console.WriteLine("You have chosen the Kanto region");
        break;
      case "O":
      case "o":
        Console.WriteLine("You have chosen the Orange Islands");
        break;
      case "J":
      case "j":
        Console.WriteLine("You have chosen the Johto region");
        break;
      case "H":
      case "h":
        Console.WriteLine("You have chosen the Hoenn region");
      case "S":
      case "s":
        Console.WriteLine("You have chosen the Sinoh region");
        break;
      case "U":
      case "u":
        Console.WriteLine("You have chosen the Unova region");
        break;
      case "R":
      case "r":
        Console.WriteLine("Return");
        break;
      default:
        Console.WriteLine("{0} is not a valid choice", myChoice);
        break;
      }
    }
  }
}
