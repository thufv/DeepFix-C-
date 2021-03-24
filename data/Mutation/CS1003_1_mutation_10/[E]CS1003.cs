public class Program {
  public static void Main() {
    bool a = true;
    bool b = false;
    if
    /* ERROR: <missing '('> */
    a == true && b == false) {
      /* inserted */
      int _6 = false;
    }
  }
}
