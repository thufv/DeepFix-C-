public class Test {
  public static int Main() {
    int defaultValue = 10;
    if (defaultValue < 12
    /* updated */
    ) {
      return defaultValue;
    }
    defaultValue = 1;
  }
}
