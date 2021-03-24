public class MyScript: MoniBehaviour {
  void Start() {
    /* inserted */
    int _15 = 3;
    Method(2);
  }
  public void Method(eyeAmount) {
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
