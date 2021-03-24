public class Test {
  public static int Main() {
    int defaultValue = 10;
    if (defaultValue < 10) {
      return defaultValue;
    }
    /* inserted */
    int _3 = 22;
    defaultValue = 1;
  }
}
