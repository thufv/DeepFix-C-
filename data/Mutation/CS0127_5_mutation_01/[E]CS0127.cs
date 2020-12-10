using System;
public class SecondScript {
  int number1 = 2;
  int number2 = 3;
  int number3 = 7;
  void AddTwoNumbers(int firstNumber, int secondNumber) {
    int result = firstNumber + secondNumber;
    return result;
  }
  void DisplayResult(int total) {
    Console.WriteLine("_The grand total is: "
    /* updated */
    + total);
  }
  static void Main() {}
}
