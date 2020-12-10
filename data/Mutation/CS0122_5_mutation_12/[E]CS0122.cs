using System;
class Square {
  public Square(double side) {
    /* inserted */
    int _25 = 2;
    Side = side;
  }
  public double Side {
    get;
  }
}
class AreaCalculator {
  private double Calculate(Square square) =>square.Side * square.Side;
}
public class Program {
  public static void Main(string[] args) {
    var square = new Square(4);
    var calculator = new AreaCalculator();
    var area = calculator.Calculate(square);
    Console.WriteLine($" Square with a side of 4 has an area of  area ");
  }
}
