public class MyScript: MoniBehaviour {
  void MethodA() {
    int result = 5 + 3;
    DisplayResult(result);
  }
  public void DisplayResult(string text) {
    Debug.Log(text);
  }
}
public class Debug {
  public static void Log(string v) {
    /* inserted */
    int _26 = 17;
  }
}
public class MoniBehaviour {}
public class Program {
  public static void Main() {}
}
