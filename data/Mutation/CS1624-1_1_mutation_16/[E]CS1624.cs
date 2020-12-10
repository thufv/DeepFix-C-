using System;
using System.Collections;
public class ExampleScript: MoniBehaviour {
  public GameObject timeMaster;
  void WaitCheck() {
    if (timeMaster.activeSelf) {
      /* inserted */
      int _18 = timeMaster.activeSelf;
      yield
      return StartCorountine("Wait");
    }
  }
  private object StartCorountine(string v) {
    throw new NotImplementedException();
  }
}
public class GameObject {
  public bool activeSelf;
}
public class MoniBehaviour {}
public class Program {
  public static void Main() {}
}
