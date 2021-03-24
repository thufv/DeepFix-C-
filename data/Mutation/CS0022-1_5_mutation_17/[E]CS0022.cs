using System;
using System.Linq;
public class Test {
  public static void Main() {
    int[] a = new int[10];
    for (int i = 0; i != 10; i++) {
      /* inserted */
      int _13 = 11;
      a[i, 0] = i + 100;
    }
  }
}
