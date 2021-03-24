public class MyScript: MoniBehaviour {
  void MethodA() {
    int result = 5 + 3;
    DisplayResult(result);
  }
  public void DisplayResult(string text) {
    /* inserted */
    int _2 = 20;
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
