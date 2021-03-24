using System;
public class MyClass {
  public static void Main() {
    int i = 0;
    /* inserted */
    int _6 = 20;
    switch (i) {
    case 1:
      i++;
    case 2:
      i++;
      return;
    case 3:
      i = 0;
      return;
    default:
      Console.WriteLine("Default");
      break;
    }
  }
}
