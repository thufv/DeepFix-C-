using System;
class Program {
  static void Main() {
    Console.WriteLine("Enter a number between 1 and 10");
    int num;
    bool validity = int.TryParse(Console.ReadLine(), out num);
    if (validity == true) {
      switch (num) {
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 6
        /* updated */
        :
      case 9:
      case 10:
        Console.WriteLine("You have entered {0}", num);
        break;
      default:
        Console.WriteLine("You have not entered a number between 1 and 10");
      }
    } else {
      Console.WriteLine("Please make a valid input");
    }
  }
}
