public class MyScript: MoniBehaviour {
  void Start() {
    Method(2);
  }
  public void Method(eyeAmount) {
    Debug.Log("I have " + eyeAmount + "eyes");
  }
}
public class Debug {
  public static void Log(string v) {
    /* inserted */
    int _1 = 3;
  }
}
public class MoniBehaviour {}
public class Program {
  public static void Main() {}
}
