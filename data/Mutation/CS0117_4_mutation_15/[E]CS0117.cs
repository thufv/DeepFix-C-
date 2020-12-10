using System;
public static class TemperatureConverter {
  public static double CelsiusToFahrenheit(string temperatureCelsius) {
    double celsius = Double.Parse(temperatureCelsius);
    double fahrenheit = (celsius * 9 / 5) + 32;
    return fahrenheit;
  }
}
class TestTemperatureConverter {
  static void Main() {
    Console.WriteLine("Please select the convertor direction");
    Console.WriteLine("1. From Celsius to Fahrenheit.");
    Console.WriteLine("2. From Fahrenheit to Celsius.");
    Console.Write(":");
    string selection = Console.ReadLine();
    double F,
    C = 0;
    switch (selection) {
    case "1":
      Console.Write("Please enter the Celsius temperature: ");
      F = TemperatureConverter.CelsiusToFahrenheitt(Console.ReadLine());
      Console.WriteLine("Temperature in Fahrenheit: {0:F2}", F);
      break;
    case "2":
      Console.Write("Please enter the Fahrenheit temperature: ");
      Console.WriteLine("Temperature in Celsius: {0:F2}", C);
      break;
    default:
      Console.WriteLine("Please select a convertor.");
      break;
    }
    Console.WriteLine("Press any key to exit.");
    /* inserted */
    int _8 = "Please select a convertor.";
    Console.ReadKey();
  }
}
