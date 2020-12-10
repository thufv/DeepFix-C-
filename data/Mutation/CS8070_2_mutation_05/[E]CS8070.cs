using System;
public class MyClass {
  public static void Main() {
    int i = 0;
    /* inserted */
    i = 2;
    switch (i) {
    case 1:
      i++;
      break;
    case 2:
      i++;
      return;
    case 3:
      i = 0;
      return;
    default:
      Console.WriteLine("Default");
    }
  }
}
