using System;
using System.Linq;
public class Test {
  public static void Main() {
    int[] a = new int[10];
    for (int i = 0; i != 10; i--
    /* updated */
    ) {
      a[i, 0] = i + 100;
    }
  }
}
