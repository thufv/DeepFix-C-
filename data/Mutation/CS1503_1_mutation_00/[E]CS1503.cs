public class MyScript: MoniBehaviour {
  void MethodA() {
    int result = 7
    /* updated */
    + 3;
    DisplayResult(result);
  }
  public void DisplayResult(string text) {
    Debug.Log(text);
  }
}
public class Debug {
  public static void Log(string v) {}
}
public class MoniBehaviour {}
public class Program {
  public static void Main() {}
}
