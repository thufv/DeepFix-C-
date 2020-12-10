public class MyScript: MoniBehaviour {
  void Start() {
    Method(2);
  }
  public void Method(eyeAmount) {
    /* inserted */
    int _15 = 8;
    Debug.Log("I have " + eyeAmount + "eyes");
  }
}
public class Debug {
  public static void Log(string v) {}
}
public class MoniBehaviour {}
public class Program {
  public static void Main() {}
}
