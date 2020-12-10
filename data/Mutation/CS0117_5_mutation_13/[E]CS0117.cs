using System;
public static class TemperatureConverter {
  public static double FahrenheitToCelsius(string temperatureFahrenheit) {
    double fahrenheit = Double.Parse(temperatureFahrenheit);
    double celsius = (fahrenheit - 32) * 5 / 9;
    return celsius;
  }
}
class TestTemperatureConverter {
  static void Main() {
    Console.WriteLine("Please select the convertor direction");
    Console.WriteLine("1. From Celsius to Fahrenheit.");
    Console.WriteLine("2. From Fahrenheit to Celsius.");
    Console.Write(":");
    string selection = Console.ReadLine();
    double F = 0,
    C = 0;
    switch (selection) {
    case "1":
      Console.Write("Please enter the Celsius temperature: ");
      Console.WriteLine("Temperature in Fahrenheit: {0:F2}", F);
      break;
    case "2":
      Console.Write("Please enter the Fahrenheit temperature: ");
      C = TemperatureConverter.FahrenheitToCelsiu(Console.ReadLine());
      Console.WriteLine("Temperature in Celsius: {0:F2}", C);
      break;
    default:
      Console.WriteLine("Please select a convertor.");
      break;
    }
    Console.WriteLine("Press any key to exit.");
    /* inserted */
    int _4 = "Please enter the Fahrenheit temperature: ";
    Console.ReadKey();
  }
}
