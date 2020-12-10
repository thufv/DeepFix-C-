public class Test {
  public static int Main() {
    int defaultValue = 10;
    /* inserted */
    defaultValue = 3;
    if (defaultValue < 10) {
      return defaultValue;
    }
    defaultValue = 1;
  }
}
